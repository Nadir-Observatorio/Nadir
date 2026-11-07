"""Land cover transitions of burned and never-burned land, tabulated by H3 cell.

For each fire year t and each pixel, the land cover class is read at chosen offsets from t (for
example t - 1, t + 2 and t + 3: before the fire, and two consecutive years after it). The area of
every combination of classes is added up by cell, separately for three groups:

- `quemado`: burned in year t and in no other year of the fire record;
- `requemado`: burned in year t and in at least one other year;
- `control`: never burned in the fire record, measured in the same years as the burned pixels.

A pixel enters only if it is inside the cell map, inside the fire export, and has a valid land
cover code in every year read (not 0, not 255, not 27 "not observed"). Classes are kept as the
original legend codes; the level-1 class of each code is added beside it.

With slope bands (from `capas.elevacion.construir_pendiente`), each row also carries the band of
its pixels, so control strata (cell, previous class and slope band) can be formed from the table.
Pixels without a slope band are then left out, and their area is reported.

Cells come from a resolution-8 map. A coarser resolution adds up child cells, so tables nest
exactly across scales.

Every area here is a result of the study once land cover meets fire. These tables are meant to be
built only after the study protocol is sealed.
"""

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path

import h3.api.numpy_int as h3n
import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq

from nadir import celdas, exportacion
from nadir.capas import cobertura, elevacion, fuego
from nadir.errors import LayerError, TableError

GRUPOS = ("control", "quemado", "requemado")
TRAMOS = ("menos_de_10", "10_a_menos_de_25", "25_o_mas")
_SIN_PENDIENTE = 3
"""Band code in the key when the table is built without slope."""
NO_OBSERVADO = 27
FRANJA = 512

_BITS = 8
_MASCARA = (1 << _BITS) - 1


@dataclass(frozen=True)
class Tabla:
    """Transition table: one row per cell, fire year, group and combination of classes."""

    resolucion: int
    desfases: tuple[int, ...]
    celdas: npt.NDArray[np.int64]
    anio: npt.NDArray[np.int32]
    grupo: npt.NDArray[np.int8]
    clases: npt.NDArray[np.uint8]
    area_m2: npt.NDArray[np.float64]
    tramo: npt.NDArray[np.int8]
    con_pendiente: bool = False
    excluido_sin_pendiente_m2: float = 0.0

    def __len__(self) -> int:
        return int(self.area_m2.size)


def _comprobar(desfases: tuple[int, ...], anios: tuple[int, ...]) -> None:
    if not desfases or len(set(desfases)) != len(desfases) or len(desfases) > 6:
        raise LayerError(f"offsets must be 1 to 6 distinct years, not {desfases}")
    if not anios or len(set(anios)) != len(anios):
        raise LayerError(f"fire years must be distinct and at least one, not {anios}")
    for anio in anios:
        fuego.banda_anual(anio)
        for desfase in desfases:
            leido = anio + desfase
            if leido not in cobertura.ANIOS_ESTUDIO:
                raise LayerError(
                    f"fire year {anio} with offset {desfase:+d} needs land cover {leido}, "
                    f"outside {cobertura.ANIOS_ESTUDIO[0]}-{cobertura.ANIOS_ESTUDIO[-1]}"
                )


def tabular(
    mapa: celdas.MapaCeldas,
    cob: exportacion.Mosaico,
    anual: exportacion.Mosaico,
    anios: tuple[int, ...],
    desfases: tuple[int, ...],
    resolucion: int = 7,
    alto: int = FRANJA,
    pendiente: Path | None = None,
) -> Tabla:
    """Area of each combination of classes at `desfases` from each fire year, by cell and group.

    With `pendiente` (a folder written by `construir_pendiente` on the same grid), rows are also
    split by slope band.
    """
    _comprobar(desfases, anios)
    if not cob.grilla == anual.grilla == mapa.grilla:
        raise LayerError("land cover, fire and cell map must share one grid")
    if alto < 1:
        raise LayerError(f"strip height must be positive, not {alto}")
    anios_cob = sorted({a + d for a in anios for d in desfases})
    n_celdas = mapa.celdas.size
    partes: dict[int, dict[int, float]] = {a: {} for a in anios}
    excluido = 0.0
    for fila in range(0, mapa.grilla.alto, alto):
        filas = min(alto, mapa.grilla.alto - fila)
        indices = mapa.indices(fila, filas)
        base = indices != celdas.SIN_CELDA
        if not np.any(base):
            continue
        quemas = {a: anual.leer(fuego.banda_anual(a), fila, filas) for a in fuego.ANIOS}
        dentro = base & (quemas[fuego.ANIOS[0]] != fuego.FUERA)
        veces = np.zeros(indices.shape, dtype=np.uint8)
        for datos in quemas.values():
            veces += (datos == fuego.QUEMADO).astype(np.uint8)
        clases = {a: cob.leer(cobertura.banda(a), fila, filas) for a in anios_cob}
        area = np.broadcast_to(mapa.grilla.area_filas_m2(fila, filas)[:, None], indices.shape)
        if pendiente is None:
            tramos = np.full(indices.shape, _SIN_PENDIENTE, dtype=np.uint8)
            con_tramo = np.ones(indices.shape, dtype=bool)
        else:
            tramos = elevacion.leer_tramos(pendiente, mapa.grilla, fila, filas)
            con_tramo = tramos != elevacion.SIN_TRAMO
        for anio in anios:
            leidas = [clases[anio + d] for d in desfases]
            valido = dentro.copy()
            for c in leidas:
                valido &= (c != cobertura.SIN_DATO) & (c != cobertura.FUERA) & (c != NO_OBSERVADO)
            grupo = np.full(indices.shape, -1, dtype=np.int8)
            quemado = quemas[anio] == fuego.QUEMADO
            grupo[veces == 0] = 0
            grupo[quemado & (veces == 1)] = 1
            grupo[quemado & (veces >= 2)] = 2
            usar = valido & (grupo >= 0)
            excluido += float(np.sum(area[usar & ~con_tramo]))
            usar &= con_tramo
            if not np.any(usar):
                continue
            clave = indices[usar].astype(np.int64)
            clave = (clave << 2) | grupo[usar].astype(np.int64)
            clave = (clave << 2) | tramos[usar].astype(np.int64)
            for c in leidas:
                clave = (clave << _BITS) | c[usar].astype(np.int64)
            unicas, inversa = np.unique(clave, return_inverse=True)
            sumas = np.bincount(inversa, weights=area[usar])
            acumulado = partes[anio]
            for k, v in zip(unicas.tolist(), sumas.tolist(), strict=True):
                acumulado[k] = acumulado.get(k, 0.0) + v
    tabla = _armar(mapa, partes, desfases, resolucion, n_celdas)
    return replace(tabla, con_pendiente=pendiente is not None, excluido_sin_pendiente_m2=excluido)


def _armar(
    mapa: celdas.MapaCeldas,
    partes: dict[int, dict[int, float]],
    desfases: tuple[int, ...],
    resolucion: int,
    n_celdas: int,
) -> Tabla:
    lista_celdas, posicion = mapa.padres(resolucion)
    n = len(desfases)
    cola = 4 + _BITS * n
    filas_anio, filas_clave, filas_area = [], [], []
    for anio, acumulado in partes.items():
        if not acumulado:
            continue
        claves = np.fromiter(acumulado, dtype=np.int64, count=len(acumulado))
        areas = np.fromiter(acumulado.values(), dtype=np.float64, count=len(acumulado))
        fina = claves >> cola
        if np.any(fina >= n_celdas):
            raise TableError("cell index outside the map")
        resto = claves & ((1 << cola) - 1)
        gruesa = posicion[fina] << cola | resto
        unicas, inversa = np.unique(gruesa, return_inverse=True)
        filas_clave.append(unicas)
        filas_area.append(np.bincount(inversa, weights=areas))
        filas_anio.append(np.full(unicas.size, anio, dtype=np.int32))
    if not filas_clave:
        vacio = np.zeros(0, dtype=np.int64)
        return Tabla(
            resolucion,
            desfases,
            vacio,
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.int8),
            np.zeros((0, n), dtype=np.uint8),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.int8),
        )
    claves = np.concatenate(filas_clave)
    anio_fuego = np.concatenate(filas_anio)
    area = np.concatenate(filas_area)
    clases = np.stack(
        [(claves >> (_BITS * (n - 1 - i))) & _MASCARA for i in range(n)], axis=1
    ).astype(np.uint8)
    tramo = ((claves >> (_BITS * n)) & 3).astype(np.int8)
    grupo = ((claves >> (_BITS * n + 2)) & 3).astype(np.int8)
    celda = lista_celdas[claves >> cola]
    llaves = [clases[:, i] for i in reversed(range(n))]
    orden = np.lexsort((*llaves, tramo, grupo, anio_fuego, celda))
    return Tabla(
        resolucion,
        desfases,
        celda[orden],
        anio_fuego[orden],
        grupo[orden],
        clases[orden],
        area[orden],
        tramo[orden],
    )


def huella(tabla: Tabla) -> str:
    """Content hash over the sorted rows, with exact float representation."""
    resumen = hashlib.sha256()
    cabecera = f"{tabla.resolucion}\t{list(tabla.desfases)}\t{tabla.con_pendiente}\n"
    resumen.update(cabecera.encode("ascii"))
    for i in range(len(tabla)):
        clases = ",".join(str(int(c)) for c in tabla.clases[i])
        linea = (
            f"{h3n.int_to_str(int(tabla.celdas[i]))}\t{int(tabla.anio[i])}\t"
            f"{GRUPOS[int(tabla.grupo[i])]}\t{int(tabla.tramo[i])}\t{clases}\t"
            f"{float(tabla.area_m2[i]).hex()}\n"
        )
        resumen.update(linea.encode("ascii"))
    return resumen.hexdigest()


def _nombre(desfase: int) -> str:
    return f"t{desfase:+d}".replace("+", "mas").replace("-", "menos")


def escribir(ruta: Path, tabla: Tabla, procedencia: dict[str, str]) -> str:
    """Write the table as Parquet with its offsets, resolution and inputs; return its hash.

    Columns: celda, anio_fuego, grupo, tramo_pendiente (only with slope), then for each offset
    the original code and its level-1 class (for example `clase_tmenos1` and `nivel1_tmenos1`),
    and area_m2. Refuses to overwrite.
    """
    if ruta.exists():
        raise TableError(f"refusing to overwrite {ruta}")
    columnas: dict[str, pa.Array] = {
        "celda": pa.array([h3n.int_to_str(int(c)) for c in tabla.celdas], type=pa.string()),
        "anio_fuego": pa.array(tabla.anio, type=pa.int32()),
        "grupo": pa.array([GRUPOS[int(g)] for g in tabla.grupo], type=pa.string()),
    }
    if tabla.con_pendiente:
        columnas["tramo_pendiente"] = pa.array(
            [TRAMOS[int(t)] for t in tabla.tramo], type=pa.string()
        )
    for i, desfase in enumerate(tabla.desfases):
        codigos = tabla.clases[:, i]
        columnas[f"clase_{_nombre(desfase)}"] = pa.array(codigos, type=pa.uint8())
        columnas[f"nivel1_{_nombre(desfase)}"] = pa.array(
            [cobertura.NIVEL_1.get(int(c), "fuera_de_leyenda") for c in codigos],
            type=pa.string(),
        )
    columnas["area_m2"] = pa.array(tabla.area_m2, type=pa.float64())
    contenido = huella(tabla)
    metadatos = {
        "resolucion": str(tabla.resolucion),
        "desfases": json.dumps(list(tabla.desfases)),
        "con_pendiente": json.dumps(tabla.con_pendiente),
        "excluido_sin_pendiente_m2": repr(tabla.excluido_sin_pendiente_m2),
        "huella": contenido,
        "procedencia": json.dumps(procedencia, sort_keys=True, ensure_ascii=False),
    }
    ruta.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(columnas).replace_schema_metadata(metadatos), ruta)
    return contenido
