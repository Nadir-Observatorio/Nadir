"""Area tables of the fire and land cover study, by H3 cell of resolution 7.

Groups follow `nadir.transiciones`: control = never burned in 2013-2025; burned = burned in the
fire year t and in no other year; reburned = burned in t and in at least one other year. A pixel
enters when it has a cell, lies inside the fire export, has a slope band and holds a valid land
cover code (not 0, not 255, not 27) in every year a table reads.

Tables (the area of a resolution-7 cell is the sum of its resolution-8 children):

- `k0`: class in t - 1, fire years 2013-2025.
- `k2`, `k3`, `k5`: classes in t - 1, t + k - 1 and t + k, and the interface band of t - 1
  (0 = less than 2 km, 1 = 2 km or more); fire years up to 2022, 2021 and 2019. In `k2` the
  band is 0 when t - 1 = 2021, a year without an interface layer (`k2` is descriptive).
- `balance`: area, area times altitude and area times distance to the interface (and their
  squares), by cell, year, group, slope band and previous level-1 class, with the validity of
  `k3`.

Each table records in its metadata the area left out, summed over fire years, for lack of a slope
band and for an invalid land cover code.

With a scar layer (`nadir.capas.lfs`), pixels carrying any of `excluir_bits` leave every group,
and pixels carrying any of `excluir_bits_control` leave the control only; the area left out is
reported per table.

Areas are accumulated strip by strip, in a fixed order, so the same inputs give the same bits.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq

from nadir import celdas, exportacion
from nadir.capas import (
    cobertura,
    cobertura_c1,
    elevacion,
    fuego,
    gabam,
    interfaz,
    interfaz_wui,
    lfs,
)
from nadir.errors import GridError, SourceError

NO_OBSERVADO = 27
FRANJA = 512
RESOLUCION = 7
TABLAS: dict[str, tuple[tuple[int, ...], range]] = {
    "k0": ((-1,), range(2013, 2026)),
    "k2": ((-1, 1, 2), range(2013, 2023)),
    "k3": ((-1, 2, 3), range(2013, 2022)),
    "k5": ((-1, 4, 5), range(2013, 2020)),
}
BALANCE = "balance"
# Exploratory: classes in t - 1, t + 4 and t + 5, the interface band and a mark for pixels that are
# Infrastructure (code 24) in every year t to t + 3, the window the Fire thematic filter masks.
EXPLORATORIAS: dict[str, tuple[tuple[int, ...], range]] = {"u5": ((-1, 4, 5), range(2013, 2020))}
# Exploratory with the fire defined by Landscape Fire Scars: seasons s = 2014-2018, classes in
# s - 2, s + 4 and s + 5; burned = inside exactly one scar of 1985-2018, of season s; control = no
# scar of 1985-2018 and never burned in Fire 2013-2025; interface band of s - 2.
EXPLORATORIAS_LFS: dict[str, tuple[tuple[int, ...], range]] = {
    "l5": ((-2, 4, 5), range(2014, 2019))
}
# H3' of the interface module: k3 with the fire taken from GABAM, years 2013-2021. Burned = burned
# in GABAM in t (group 1 once in 2013-2021, group 2 more than once); control = never burned in
# GABAM 2013-2021 and never burned in Fire 2013-2025; interface band of t - 1.
CONFIRMATORIAS_GABAM: dict[str, tuple[tuple[int, ...], range]] = {
    "g3": ((-1, 2, 3), range(2013, 2022))
}
INFRAESTRUCTURA = 24
# Sensitivity 7: k3 rebuilt with land cover collection 1, fire years 2013-2019.
COLECCIONES = (1, 2)
TABLA_C1 = "k3"
_CONOCIDO_C1 = np.zeros(256, dtype=bool)
_CONOCIDO_C1[[*cobertura_c1.NIVEL_1, cobertura.SIN_DATO, cobertura.FUERA]] = True
VENTANA_FILTRO = (0, 1, 2, 3)
NIVEL1_INDICE = {
    "formacion_boscosa": 1,
    "formacion_natural_no_boscosa": 2,
    "agropecuaria_y_silvicultura": 3,
    "area_sin_vegetacion": 4,
    "cuerpo_de_agua": 5,
}

Arreglo = npt.NDArray[np.float64]
Claves = npt.NDArray[np.int64]


def _tabla_nivel1() -> npt.NDArray[np.uint8]:
    tabla = np.zeros(256, dtype=np.uint8)
    for codigo, nombre in cobertura.NIVEL_1.items():
        tabla[codigo] = NIVEL1_INDICE.get(nombre, 0)
    return tabla


NIVEL1 = _tabla_nivel1()


@dataclass(frozen=True)
class Insumos:
    """Folders of the derived layers the tables read, and the scar exclusion if any."""

    celdas: Path
    pendiente: Path
    interfaz: Path
    cicatrices: Path | None = None
    excluir_bits: int = 0
    excluir_bits_control: int = 0
    conteo_lfs: Path | None = None
    interfaz_wui: Path | None = None
    gabam: Path | None = None

    def __post_init__(self) -> None:
        if (self.excluir_bits or self.excluir_bits_control) and self.cicatrices is None:
            raise SourceError("excluding scar flags needs the scar layer folder")
        if not 0 <= self.excluir_bits <= 255 or not 0 <= self.excluir_bits_control <= 255:
            raise SourceError("scar flags are one byte")


def _sumar(
    partes: list[tuple[Claves, Arreglo, Arreglo]],
) -> tuple[Claves, Arreglo, Arreglo]:
    claves = np.concatenate([p[0] for p in partes])
    unicas, inversa = np.unique(claves, return_inverse=True)
    area = np.bincount(inversa, weights=np.concatenate([p[1] for p in partes]))
    pixeles = np.bincount(inversa, weights=np.concatenate([p[2] for p in partes]))
    return unicas, area, pixeles


def _sumar_balance(partes: list[tuple[Claves, Arreglo]]) -> tuple[Claves, Arreglo]:
    claves = np.concatenate([p[0] for p in partes])
    unicas, inversa = np.unique(claves, return_inverse=True)
    sumas = np.stack(
        [
            np.bincount(inversa, weights=np.concatenate([p[1][:, j] for p in partes]))
            for j in range(5)
        ],
        axis=1,
    )
    return unicas, sumas


def _comprobar_grillas(cob: exportacion.Grilla, anual: exportacion.Grilla) -> None:
    if not exportacion.misma_grilla(cob, anual):
        raise GridError("land cover and fire exports are not on the same grid")


def construir(
    bruto: Path,
    insumos: Insumos,
    salida: Path,
    elegidas: tuple[str, ...] = (*TABLAS, BALANCE),
    coleccion: int = 2,
) -> dict[str, str]:
    """Build the chosen tables into `salida` and return the content hash of each.

    With a scar exclusion, the area left out of each table is saved in its metadata.

    `balance` needs `k3` among the chosen tables. Never overwrites.

    With `coleccion=1` the classes come from land cover collection 1 (sensitivity 7): only `k3`,
    fire years 2013-2019, on the grid of collection 2; the interface band still comes from the
    interface layer of collection 2. A code outside the level-1 table of collection 1 stops it.
    """
    if coleccion not in COLECCIONES:
        raise SourceError(f"land cover collections are {COLECCIONES}, not {coleccion}")
    todas = {**TABLAS, **EXPLORATORIAS, **EXPLORATORIAS_LFS, **CONFIRMATORIAS_GABAM}
    desconocidas = sorted(set(elegidas) - {*todas, BALANCE})
    if desconocidas:
        raise SourceError(f"unknown tables: {desconocidas}")
    tablas = {k: v for k, v in todas.items() if k in elegidas}
    if coleccion == 1:
        if tuple(elegidas) != (TABLA_C1,):
            raise SourceError(f"with collection 1 only {TABLA_C1} is built (sensitivity 7)")
        tablas = {TABLA_C1: (TABLAS[TABLA_C1][0], cobertura_c1.ANIOS_FUEGO)}
    if insumos.interfaz_wui is not None and tuple(elegidas) != ("k3",):
        raise SourceError("with the WUI band only k3 is built (H4 sensitivity)")
    if any(n in CONFIRMATORIAS_GABAM for n in tablas) and insumos.gabam is None:
        raise SourceError("the g3 table needs the GABAM layer (gabam construir)")
    if any(n in EXPLORATORIAS_LFS for n in tablas) and insumos.conteo_lfs is None:
        raise SourceError("the l5 table needs the scar count layer (lfs conteo)")
    con_balance = BALANCE in elegidas
    if con_balance and "k3" not in tablas:
        raise SourceError("the balance table is built together with k3")
    for nombre in [*tablas, *([BALANCE] if con_balance else [])]:
        if (salida / f"{nombre}.parquet").exists():
            raise SourceError(f"refusing to overwrite {salida / nombre}.parquet")
    salida.mkdir(parents=True, exist_ok=True)
    cob = cobertura_c1.abrir(bruto) if coleccion == 1 else cobertura.abrir(bruto)
    anual, mensual = fuego.abrir(bruto)
    mensual.cerrar()
    try:
        g = cob.grilla
        _comprobar_grillas(g, anual.grilla)
        mapa = celdas.abrir(insumos.celdas, g)
        lista7, pos7 = mapa.padres(RESOLUCION)
        acum, bal, fuera, mascaras = _acumular(
            cob, anual, mapa, pos7, insumos, tablas, con_balance, coleccion
        )
    finally:
        cob.cerrar()
        anual.cerrar()
    huellas = {}
    for nombre, (desfases, _) in tablas.items():
        huellas[nombre] = _escribir_tabla(
            salida,
            nombre,
            desfases,
            acum[nombre][0],
            lista7,
            {
                **_exclusion(insumos, fuera[nombre]),
                **_mascaras(mascaras[nombre]),
                **({"coleccion": "1"} if coleccion == 1 else {}),
                **({"interfaz": "wui"} if insumos.interfaz_wui is not None else {}),
                **({"fuego": "gabam"} if nombre in CONFIRMATORIAS_GABAM else {}),
            },
        )
    if con_balance:
        huellas[BALANCE] = _escribir_balance(
            salida,
            bal[0],
            lista7,
            {**_exclusion(insumos, fuera["k3"]), **_mascaras(mascaras["k3"])},
        )
    return huellas


def _acumular(
    cob: exportacion.Mosaico,
    anual: exportacion.Mosaico,
    mapa: celdas.MapaCeldas,
    pos7: Claves,
    insumos: Insumos,
    tablas: dict[str, tuple[tuple[int, ...], range]],
    con_balance: bool,
    coleccion: int = 2,
) -> tuple[
    dict[str, list[tuple[Claves, Arreglo, Arreglo]]],
    list[tuple[Claves, Arreglo]],
    dict[str, float],
    dict[str, dict[str, float]],
]:
    g = cob.grilla
    acum: dict[str, list[tuple[Claves, Arreglo, Arreglo]]] = {k: [] for k in tablas}
    fuera = dict.fromkeys(tablas, 0.0)
    mascaras = {k: {"sin_tramo": 0.0, "codigo_no_valido": 0.0} for k in tablas}
    excluye = bool(insumos.excluir_bits or insumos.excluir_bits_control)
    bal: list[tuple[Claves, Arreglo]] = []
    anios_cob = sorted({a + d for ds, ans in tablas.values() for a in ans for d in ds})
    for nombre in tablas:
        if nombre in EXPLORATORIAS:
            ventana = {a + d for a in tablas[nombre][1] for d in VENTANA_FILTRO}
            anios_cob = sorted({*anios_cob} | ventana)
    for fila in range(0, g.alto, FRANJA):
        filas = min(FRANJA, g.alto - fila)
        idx8 = mapa.indices(fila, filas)
        base = idx8 != celdas.SIN_CELDA
        idx = np.zeros(idx8.shape, dtype=np.int64)
        idx[base] = pos7[idx8[base].astype(np.int64)]
        if not base.any():
            continue
        quemas = {a: anual.leer(fuego.banda_anual(a), fila, filas) for a in fuego.ANIOS}
        dentro = base & (quemas[fuego.ANIOS[0]] != fuego.FUERA)
        veces = np.zeros(idx.shape, dtype=np.uint8)
        for q in quemas.values():
            veces += (q == fuego.QUEMADO).astype(np.uint8)
        tram = elevacion.leer_tramos(insumos.pendiente, g, fila, filas)
        sin_tramo = dentro & (tram == elevacion.SIN_TRAMO)
        dentro &= tram != elevacion.SIN_TRAMO
        nombre_banda = cobertura_c1.banda if coleccion == 1 else cobertura.banda
        clases = {a: cob.leer(nombre_banda(a), fila, filas) for a in anios_cob}
        if coleccion == 1:
            for a, c in clases.items():
                extranos = np.unique(c[base & ~_CONOCIDO_C1[c]])
                if extranos.size:
                    raise SourceError(
                        f"{a}: codes outside the level-1 table of collection 1: "
                        f"{extranos.tolist()}"
                    )
        area = np.broadcast_to(g.area_filas_m2(fila, filas)[:, None], idx.shape)
        if excluye and insumos.cicatrices is not None:
            banderas = lfs.leer_banderas(insumos.cicatrices, g, fila, filas)
            fuera_todos = (banderas & insumos.excluir_bits) != 0
            fuera_control = (banderas & insumos.excluir_bits_control) != 0
        if insumos.gabam is not None and any(n in CONFIRMATORIAS_GABAM for n in tablas):
            quemas_g = {a: gabam.leer(insumos.gabam, g, a, fila, filas) for a in gabam.ANIOS}
            veces_g = np.zeros(idx.shape, dtype=np.uint8)
            fuera_g = np.zeros(idx.shape, dtype=bool)
            for q in quemas_g.values():
                veces_g += (q == gabam.QUEMADO).astype(np.uint8)
                fuera_g |= q == gabam.FUERA
        if insumos.conteo_lfs is not None and any(n in EXPLORATORIAS_LFS for n in tablas):
            conteo, temporada_unica = lfs.leer_conteo(insumos.conteo_lfs, g, fila, filas)
        altura: Arreglo | None = None
        if con_balance:
            altura = np.asarray(
                elevacion.leer_altitud(insumos.pendiente, g, fila, filas), dtype=np.float64
            )
        for nombre, (desfases, anios) in tablas.items():
            for anio in anios:
                leidas = [clases[anio + d] for d in desfases]
                valido = dentro.copy()
                for c in leidas:
                    valido &= (
                        (c != cobertura.SIN_DATO) & (c != cobertura.FUERA) & (c != NO_OBSERVADO)
                    )
                grupo = np.full(idx.shape, -1, dtype=np.int8)
                if nombre in CONFIRMATORIAS_GABAM:
                    quem_g = (quemas_g[anio] == gabam.QUEMADO) & ~fuera_g
                    grupo[(veces_g == 0) & (veces == 0) & ~fuera_g] = 0
                    grupo[quem_g & (veces_g == 1)] = 1
                    grupo[quem_g & (veces_g >= 2)] = 2
                elif nombre in EXPLORATORIAS_LFS:
                    grupo[(conteo == 0) & (veces == 0)] = 0
                    grupo[(conteo == 1) & (temporada_unica == anio - lfs.TEMPORADA_BASE)] = 1
                else:
                    quem = quemas[anio] == fuego.QUEMADO
                    grupo[veces == 0] = 0
                    grupo[quem & (veces == 1)] = 1
                    grupo[quem & (veces >= 2)] = 2
                usar = valido & (grupo >= 0)
                en_grupo = grupo >= 0
                mascaras[nombre]["sin_tramo"] += float(np.sum(area[sin_tramo & en_grupo]))
                mascaras[nombre]["codigo_no_valido"] += float(
                    np.sum(area[dentro & en_grupo & ~valido])
                )
                if excluye:
                    sacar = fuera_todos | (fuera_control & (grupo == 0))
                    fuera[nombre] += float(np.sum(area[usar & sacar]))
                    usar &= ~sacar
                if not usar.any():
                    continue
                clave = idx[usar].astype(np.int64)
                clave = (clave << 4) | anio - 2010
                clave = (clave << 2) | grupo[usar].astype(np.int64)
                clave = (clave << 2) | tram[usar].astype(np.int64)
                if nombre != "k0":
                    previo = anio - 2 if nombre in EXPLORATORIAS_LFS else anio - 1
                    clave = (clave << 1) | _banda(insumos, g, previo, fila, filas, usar)
                if nombre in EXPLORATORIAS:
                    marca = np.ones(idx.shape, dtype=bool)
                    for d in VENTANA_FILTRO:
                        marca &= clases[anio + d] == INFRAESTRUCTURA
                    clave = (clave << 1) | marca[usar].astype(np.int64)
                for c in leidas:
                    clave = (clave << 8) | c[usar].astype(np.int64)
                u, inv = np.unique(clave, return_inverse=True)
                acum[nombre].append(
                    (u, np.bincount(inv, weights=area[usar]), np.bincount(inv).astype(np.float64))
                )
                if nombre == "k3" and altura is not None:
                    bal.append(
                        _parte_balance(insumos, g, anio, fila, filas, usar, idx, grupo, tram,
                                       leidas[0], area, altura)
                    )
        for nombre in acum:
            if acum[nombre]:
                acum[nombre] = [_sumar(acum[nombre])]
        if bal:
            bal = [_sumar_balance(bal)]
    return acum, bal, fuera, mascaras


def _mascaras(areas: dict[str, float]) -> dict[str, str]:
    return {
        "excluido_sin_tramo_m2": repr(areas["sin_tramo"]),
        "excluido_codigo_no_valido_m2": repr(areas["codigo_no_valido"]),
    }


def _exclusion(insumos: Insumos, area_fuera: float) -> dict[str, str]:
    if not (insumos.excluir_bits or insumos.excluir_bits_control):
        return {}
    return {
        "excluir_bits": str(insumos.excluir_bits),
        "excluir_bits_control": str(insumos.excluir_bits_control),
        "excluido_por_cicatrices_m2": repr(area_fuera),
    }


def _banda(
    insumos: Insumos,
    g: exportacion.Grilla,
    previo: int,
    fila: int,
    filas: int,
    usar: npt.NDArray[np.bool_],
) -> Claves:
    if insumos.interfaz_wui is not None:
        banda = interfaz_wui.leer_banda(insumos.interfaz_wui, g, fila, filas)[usar]
        if np.any(banda == interfaz_wui.FUERA):
            raise SourceError("WUI band missing for pixels in the study")
        return banda.astype(np.int64)
    if previo not in interfaz.ANIOS:
        return np.zeros(int(usar.sum()), dtype=np.int64)
    banda = interfaz.leer_banda(insumos.interfaz, g, previo, fila, filas)[usar]
    if np.any(banda == interfaz.FUERA):
        raise SourceError(f"interface band of {previo} missing for pixels in the study")
    return banda.astype(np.int64)


def _parte_balance(
    insumos: Insumos,
    g: exportacion.Grilla,
    anio: int,
    fila: int,
    filas: int,
    usar: npt.NDArray[np.bool_],
    idx: Claves,
    grupo: npt.NDArray[np.int8],
    tram: npt.NDArray[np.uint8],
    previa: npt.NDArray[np.uint8],
    area: Arreglo,
    altura: Arreglo,
) -> tuple[Claves, Arreglo]:
    dist = interfaz.leer_dist40(insumos.interfaz, g, anio - 1, fila, filas)
    dm = np.asarray(dist, dtype=np.float64)[usar] * interfaz.PASO_SALIDA_M
    h = altura[usar]
    w = area[usar]
    nivel1 = NIVEL1[previa[usar]].astype(np.int64)
    clave = idx[usar].astype(np.int64)
    clave = (clave << 4) | anio - 2010
    clave = (clave << 2) | grupo[usar].astype(np.int64)
    clave = (clave << 2) | tram[usar].astype(np.int64)
    clave = (clave << 3) | nivel1
    u, inv = np.unique(clave, return_inverse=True)
    columnas = [w, w * h, w * h * h, w * dm, w * dm * dm]
    sumas = np.stack([np.bincount(inv, weights=c) for c in columnas], axis=1)
    return u, sumas


def _escribir_tabla(
    salida: Path,
    nombre: str,
    desfases: tuple[int, ...],
    acumulado: tuple[Claves, Arreglo, Arreglo],
    lista7: Claves,
    extra: dict[str, str],
) -> str:
    claves, area, pixeles = acumulado
    columnas: dict[str, npt.NDArray[np.generic]] = {}
    resto = claves
    for i in reversed(range(len(desfases))):
        columnas[f"c{i}"] = (resto & 255).astype(np.uint8)
        resto = resto >> 8
    if nombre in EXPLORATORIAS:
        columnas["m24"] = (resto & 1).astype(np.uint8)
        resto = resto >> 1
    if nombre != "k0":
        columnas["banda"] = (resto & 1).astype(np.uint8)
        resto = resto >> 1
    columnas["tramo"] = (resto & 3).astype(np.uint8)
    resto = resto >> 2
    columnas["grupo"] = (resto & 3).astype(np.uint8)
    resto = resto >> 2
    columnas["anio"] = ((resto & 15) + 2010).astype(np.int16)
    resto = resto >> 4
    columnas["celda"] = lista7[resto].astype(np.int64)
    columnas["area_m2"] = area
    columnas["n_px"] = pixeles.astype(np.int64)
    orden = sorted(columnas)
    tabla = pa.table({k: columnas[k] for k in orden})
    huella = hashlib.sha256(
        b"".join(np.ascontiguousarray(tabla.column(c).to_numpy()).tobytes() for c in orden)
    ).hexdigest()
    metadatos = {"tabla": nombre, "desfases": str(list(desfases)), "huella": huella, **extra}
    pq.write_table(tabla.replace_schema_metadata(metadatos), salida / f"{nombre}.parquet")
    return huella


def _escribir_balance(
    salida: Path, acumulado: tuple[Claves, Arreglo], lista7: Claves, extra: dict[str, str]
) -> str:
    claves, sumas = acumulado
    tabla = pa.table(
        {
            "celda": lista7[claves >> 11],
            "anio": (((claves >> 7) & 15) + 2010).astype(np.int16),
            "grupo": ((claves >> 5) & 3).astype(np.uint8),
            "tramo": ((claves >> 3) & 3).astype(np.uint8),
            "nivel1_previo": (claves & 7).astype(np.uint8),
            "w": sumas[:, 0],
            "w_alt": sumas[:, 1],
            "w_alt2": sumas[:, 2],
            "w_dist": sumas[:, 3],
            "w_dist2": sumas[:, 4],
        }
    )
    huella = hashlib.sha256(sumas.tobytes() + claves.tobytes()).hexdigest()
    metadatos = {"tabla": BALANCE, "huella": huella, **extra}
    pq.write_table(tabla.replace_schema_metadata(metadatos), salida / f"{BALANCE}.parquet")
    return huella
