"""Distance to the urban interface on the native grid of the MapBiomas Chile exports.

The interface of year t is the Infrastructure code (24) of land cover in year t. Distance is
Euclidean, in metres, computed on the native EPSG:4326 grid without reprojecting or resampling:
the grid is cut into strips of at most 0.1 degrees of latitude, and each strip uses the metric
spacing at its central latitude on the WGS84 ellipsoid (meridian radius for rows, prime vertical
radius times the cosine of latitude for columns). Each strip is computed on a window that adds
`TOPE_M` above and below, so every distance up to `TOPE_M` is exact within the strip; larger
distances are kept as `TOPE_M` or more.

Two files per year, one byte per pixel, row-major, on the export grid:

- `banda_<year>.u8`: 0 = less than `CORTE_M`; 1 = `CORTE_M` or more; `FUERA` outside the export.
- `dist40_<year>.u8`: distance in steps of `PASO_SALIDA_M`, rounded, capped at `TOPE_M`
  (value 250); `FUERA` outside the export.
"""

import hashlib
import json
import math
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path

import numpy as np
import numpy.typing as npt
from scipy.ndimage import distance_transform_edt

from nadir import exportacion
from nadir.capas import cobertura
from nadir.errors import SourceError

SEMIEJE_MAYOR = 6378137.0
ACHATAMIENTO = 1 / 298.257223563
INFRAESTRUCTURA = 24
CORTE_M = 2000.0
TOPE_M = 10000.0
PASO_SALIDA_M = 40.0
TOPE_PASOS = 250
FUERA = 255
ANIOS = tuple(range(2012, 2021))
FRANJA_GRADOS = 0.1
LATITUD_MARGEN = -36.0
_REGISTRO = "registro.json"


def radios(latitud: float) -> tuple[float, float]:
    """Meridian and prime vertical radii of the WGS84 ellipsoid at this latitude, in metres."""
    e2 = ACHATAMIENTO * (2 - ACHATAMIENTO)
    seno = math.sin(math.radians(latitud))
    w = math.sqrt(1 - e2 * seno * seno)
    return SEMIEJE_MAYOR * (1 - e2) / w**3, SEMIEJE_MAYOR / w


def espaciado(latitud: float, paso: float) -> tuple[float, float]:
    """Row and column spacing in metres of a pixel of `paso` degrees at this latitude."""
    meridiano, normal = radios(latitud)
    dy = meridiano * math.radians(paso)
    dx = normal * math.cos(math.radians(latitud)) * math.radians(paso)
    return dy, dx


def margen_filas(paso: float) -> int:
    """Rows added above and below each strip so that distances up to `TOPE_M` are exact."""
    meridiano, _ = radios(LATITUD_MARGEN)
    return math.ceil(TOPE_M / (meridiano * math.radians(paso))) + 2


def distancia(
    ventana: npt.NDArray[np.uint8], dy: float, dx: float
) -> npt.NDArray[np.float64]:
    """Distance in metres from each pixel of `ventana` to the nearest Infrastructure pixel.

    Infinite everywhere when the window has no Infrastructure pixel.
    """
    interfaz = ventana == INFRAESTRUCTURA
    if not interfaz.any():
        return np.full(ventana.shape, np.inf)
    resultado: npt.NDArray[np.float64] = distance_transform_edt(~interfaz, sampling=(dy, dx))
    return resultado


def codificar(
    metros: npt.NDArray[np.float64], fuera: npt.NDArray[np.bool_]
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    """Band (0 near, 1 far) and distance in `PASO_SALIDA_M` steps, with `FUERA` outside."""
    banda = np.where(metros < CORTE_M, 0, 1).astype(np.uint8)
    banda[fuera] = FUERA
    pasos = np.minimum(np.rint(np.minimum(metros, TOPE_M) / PASO_SALIDA_M), TOPE_PASOS)
    dist40 = pasos.astype(np.uint8)
    dist40[fuera] = FUERA
    return banda, dist40


def _franja_borde(
    ventana: npt.NDArray[np.uint8],
    sin_dato: int,
    desde: int,
    filas: int,
    dy: float,
    dx: float,
    area_filas: npt.NDArray[np.float64],
) -> float:
    """Area of the strip rows that lie within `CORTE_M` of a pixel outside the export."""
    fuera = ventana == sin_dato
    if not fuera.any():
        return 0.0
    al_borde = distance_transform_edt(~fuera, sampling=(dy, dx))[desde : desde + filas]
    cerca = (al_borde < CORTE_M) & ~fuera[desde : desde + filas]
    return float((cerca * area_filas[:, None]).sum())


def _archivos(anio: int) -> tuple[str, str]:
    return f"banda_{anio}.u8", f"dist40_{anio}.u8"


def construir(
    mosaico: exportacion.Mosaico, carpeta: Path, anios: Iterable[int] = ANIOS
) -> dict[str, object]:
    """Interface band and distance of each year for every pixel of the land cover export.

    Writes into `carpeta` and never overwrites. Returns the record also saved there.
    """
    anios = tuple(anios)
    for anio in anios:
        cobertura.banda(anio)
        if any((carpeta / n).exists() for n in _archivos(anio)):
            raise SourceError(f"refusing to overwrite the interface files of {anio} in {carpeta}")
    if (carpeta / _REGISTRO).exists():
        raise SourceError(f"refusing to overwrite {carpeta / _REGISTRO}")
    carpeta.mkdir(parents=True, exist_ok=True)
    g = mosaico.grilla
    filas_franja = math.floor(FRANJA_GRADOS / g.paso)
    margen = margen_filas(g.paso)
    por_anio: dict[str, dict[str, object]] = {}
    franja_borde_m2 = 0.0
    for i, anio in enumerate(anios):
        nombre = cobertura.banda(anio)
        n_banda, n_dist = _archivos(anio)
        parciales = (carpeta / f"{n_banda}.part", carpeta / f"{n_dist}.part")
        h_banda, h_dist = hashlib.sha256(), hashlib.sha256()
        dentro = cerca = 0
        with parciales[0].open("wb") as s_banda, parciales[1].open("wb") as s_dist:
            for f0 in range(0, g.alto, filas_franja):
                n = min(filas_franja, g.alto - f0)
                v0 = max(0, f0 - margen)
                v1 = min(g.alto, f0 + n + margen)
                ventana = mosaico.leer(nombre, v0, v1 - v0)
                lat_c = float(g.latitudes(f0, n).mean())
                dy, dx = espaciado(lat_c, g.paso)
                metros = distancia(ventana, dy, dx)[f0 - v0 : f0 - v0 + n]
                fuera = ventana[f0 - v0 : f0 - v0 + n] == mosaico.sin_dato
                banda, dist40 = codificar(metros, fuera)
                if i == 0:
                    franja_borde_m2 += _franja_borde(ventana, mosaico.sin_dato, f0 - v0, n, dy, dx,
                                                     g.area_filas_m2(f0, n))
                dentro += int(np.sum(banda != FUERA))
                cerca += int(np.sum(banda == 0))
                for flujo, resumen, datos in ((s_banda, h_banda, banda), (s_dist, h_dist, dist40)):
                    bytes_ = datos.tobytes()
                    flujo.write(bytes_)
                    resumen.update(bytes_)
        parciales[0].replace(carpeta / n_banda)
        parciales[1].replace(carpeta / n_dist)
        por_anio[str(anio)] = {
            "pixeles_dentro": dentro,
            "pixeles_cerca": cerca,
            "sha256_banda": h_banda.hexdigest(),
            "sha256_dist40": h_dist.hexdigest(),
        }
    registro: dict[str, object] = {
        "grilla": asdict(g),
        "metodo": "Euclidean distance to land cover code 24 on the native grid, strips of at "
        "most 0.1 degrees with WGS84 metric spacing at the strip's central latitude",
        "corte_m": CORTE_M,
        "tope_m": TOPE_M,
        "paso_salida_m": PASO_SALIDA_M,
        "filas_por_franja": filas_franja,
        "franja_borde_2km_ha": round(franja_borde_m2 / 1e4, 1),
        "franja_borde_nota": "area within 2 km of a pixel outside the export (land border and "
        "coast alike), where distances can be overestimated",
        "margen_filas": margen,
        "anios": por_anio,
    }
    texto = json.dumps(registro, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    (carpeta / _REGISTRO).write_bytes(texto.encode("utf-8"))
    return registro


def _leer(
    carpeta: Path, grilla: exportacion.Grilla, archivo: str, fila: int, filas: int
) -> npt.NDArray[np.uint8]:
    registro = json.loads((carpeta / _REGISTRO).read_text(encoding="utf-8"))
    if not exportacion.misma_grilla(exportacion.Grilla(**registro["grilla"]), grilla):
        raise SourceError(f"the interface files in {carpeta} belong to another grid")
    with (carpeta / archivo).open("rb") as origen:
        origen.seek(fila * grilla.ancho)
        datos = np.fromfile(origen, dtype="u1", count=filas * grilla.ancho)
    if datos.size != filas * grilla.ancho:
        raise SourceError(f"{carpeta / archivo} is shorter than its grid")
    return datos.reshape(filas, grilla.ancho)


def leer_banda(
    carpeta: Path, grilla: exportacion.Grilla, anio: int, fila: int, filas: int
) -> npt.NDArray[np.uint8]:
    """Interface band of `anio` for rows `fila` to `fila + filas - 1`."""
    return _leer(carpeta, grilla, _archivos(anio)[0], fila, filas)


def leer_dist40(
    carpeta: Path, grilla: exportacion.Grilla, anio: int, fila: int, filas: int
) -> npt.NDArray[np.uint8]:
    """Distance of `anio` in `PASO_SALIDA_M` steps for rows `fila` to `fila + filas - 1`."""
    return _leer(carpeta, grilla, _archivos(anio)[1], fila, filas)
