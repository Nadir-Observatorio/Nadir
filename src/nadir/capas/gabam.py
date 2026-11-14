"""30 m Global Annual Burned Area Maps (GABAM) on the native grid of the MapBiomas exports.

Fire source of H3' of the interface module: a definition of fire that does not go through the
thematic filter of MapBiomas Fire. GABAM (Long et al., 2019) maps, for each year, the extent of the
fires of that year from Landsat; its grid is 0.00025 degrees in EPSG:4326. The seven regions are
exported from the Earth Engine community catalog with `gee/exportar_gabam.js` (one export per
year, `QUEMADO` burned, `NO_QUEMADO` not burned, `FUERA` outside the seven regions) and the zips
that Drive hands back go into the bank untouched.

Each pixel of the land cover grid takes the GABAM pixel that holds its centre; nothing is
averaged. One file per year, one byte per pixel, row-major, on the land cover grid:
`quemado_<year>.u8` with `QUEMADO`, `NO_QUEMADO` or `FUERA` (outside either export).
"""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import numpy.typing as npt

from nadir import banco, exportacion
from nadir.errors import SourceError

CARPETA = Path("gabam") / "gee-7-regiones"
ANIOS = tuple(range(2013, 2022))
NO_QUEMADO = 0
QUEMADO = 1
FUERA = 255
FILAS_FRANJA = 512
_REGISTRO = "registro_gabam.json"


def prefijo(anio: int) -> str:
    """File prefix of the export of one year."""
    return f"gabam_7_regiones_{anio}"


def banda(anio: int) -> str:
    """Band name of the export of one year."""
    return f"quemado_{anio}"


def _archivo(anio: int) -> str:
    return f"quemado_{anio}.u8"


def abrir(bruto: Path, anio: int) -> exportacion.Mosaico:
    """Open the export of one year from the bank, after checking every recorded file."""
    if anio not in ANIOS:
        raise SourceError(f"GABAM is read for {ANIOS[0]}-{ANIOS[-1]}, not {anio}")
    mosaico = exportacion.abrir(banco.zips(bruto / CARPETA), prefijo(anio))
    if mosaico.bandas != (banda(anio),) or mosaico.sin_dato != FUERA:
        mosaico.cerrar()
        raise SourceError(
            f"{prefijo(anio)}: bands {mosaico.bandas} and no-data {mosaico.sin_dato}, "
            f"expected ({banda(anio)},) and {FUERA}"
        )
    return mosaico


def muestrear(
    fuente: exportacion.Mosaico, nombre: str, destino: exportacion.Grilla, fila: int, filas: int
) -> npt.NDArray[np.uint8]:
    """Value of the source pixel that holds the centre of each pixel of `destino` in the rows."""
    g = fuente.grilla
    salida = np.full((filas, destino.ancho), FUERA, dtype=np.uint8)
    f = np.floor((g.norte - destino.latitudes(fila, filas)) / g.paso).astype(np.int64)
    c = np.floor((destino.longitudes(0, destino.ancho) - g.oeste) / g.paso).astype(np.int64)
    fok = (f >= 0) & (f < g.alto)
    cok = (c >= 0) & (c < g.ancho)
    if not fok.any() or not cok.any():
        return salida
    f0, f1 = int(f[fok].min()), int(f[fok].max())
    bloque = fuente.leer(nombre, f0, f1 - f0 + 1)
    filas_ok = np.nonzero(fok)[0]
    salida[np.ix_(filas_ok, np.nonzero(cok)[0])] = bloque[np.ix_(f[fok] - f0, c[cok])]
    return salida


def construir(
    bruto: Path, mosaico: exportacion.Mosaico, carpeta: Path, mascara: str
) -> dict[str, object]:
    """Burned band of every GABAM year for every pixel of the land cover export.

    `mascara` is a land cover band: its no-data pixels are `FUERA`. Writes into `carpeta` and
    never overwrites. Returns the record also saved there.
    """
    if (carpeta / _REGISTRO).exists() or any((carpeta / _archivo(a)).exists() for a in ANIOS):
        raise SourceError(f"refusing to overwrite the GABAM files in {carpeta}")
    archivos = {a.nombre: a for a in banco.comprobar(bruto / CARPETA)}
    carpeta.mkdir(parents=True, exist_ok=True)
    g = mosaico.grilla
    por_anio: dict[str, dict[str, object]] = {}
    for anio in ANIOS:
        resumen = hashlib.sha256()
        quemado = dentro = 0
        area_q = 0.0
        parcial = carpeta / f"{_archivo(anio)}.part"
        with abrir(bruto, anio) as fuente, parcial.open("wb") as salida:
            for f0 in range(0, g.alto, FILAS_FRANJA):
                n = min(FILAS_FRANJA, g.alto - f0)
                valor = muestrear(fuente, banda(anio), g, f0, n)
                raros = np.unique(valor[(valor != QUEMADO) & (valor != NO_QUEMADO)
                                        & (valor != FUERA)])
                if raros.size:
                    raise SourceError(f"GABAM {anio}: unexpected values {raros.tolist()}")
                valor[mosaico.leer(mascara, f0, n) == mosaico.sin_dato] = FUERA
                dentro += int((valor != FUERA).sum())
                quemado += int((valor == QUEMADO).sum())
                area_q += float(((valor == QUEMADO) * g.area_filas_m2(f0, n)[:, None]).sum())
                datos = valor.tobytes()
                salida.write(datos)
                resumen.update(datos)
            grilla_fuente = asdict(fuente.grilla)
        parcial.replace(carpeta / _archivo(anio))
        por_anio[str(anio)] = {
            "pixeles_dentro": dentro,
            "pixeles_quemados": quemado,
            "area_quemada_ha": round(area_q / 1e4, 1),
            "grilla_fuente": grilla_fuente,
            "sha256": resumen.hexdigest(),
        }
    registro: dict[str, object] = {
        "grilla": asdict(g),
        "metodo": "each land cover pixel takes the GABAM pixel that holds its centre",
        "mascara": mascara,
        "fuentes": {n: {"bytes": a.bytes, "sha256": a.sha256} for n, a in sorted(archivos.items())},
        "anios": por_anio,
    }
    texto = json.dumps(registro, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    (carpeta / _REGISTRO).write_bytes(texto.encode("utf-8"))
    return registro


def leer(
    carpeta: Path, grilla: exportacion.Grilla, anio: int, fila: int, filas: int
) -> npt.NDArray[np.uint8]:
    """Burned band of `anio` for rows `fila` to `fila + filas - 1`."""
    registro = json.loads((carpeta / _REGISTRO).read_text(encoding="utf-8"))
    if not exportacion.misma_grilla(exportacion.Grilla(**registro["grilla"]), grilla):
        raise SourceError(f"the GABAM files in {carpeta} belong to another grid")
    with (carpeta / _archivo(anio)).open("rb") as origen:
        origen.seek(fila * grilla.ancho)
        datos = np.fromfile(origen, dtype="u1", count=filas * grilla.ancho)
    if datos.size != filas * grilla.ancho:
        raise SourceError(f"{carpeta / _archivo(anio)} is shorter than its grid")
    return datos.reshape(filas, grilla.ancho)
