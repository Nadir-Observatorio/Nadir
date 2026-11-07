"""The common substrate: H3 cells at fixed resolutions (docs/design.md, section 2).

Pixels become cells by their centre: each pixel belongs to the cell that contains its centre,
weighted by its area in km². The aggregation rule comes from the variable registry.
`fraccion_valida` is the area with valid data inside the cell divided by the cell area, so a
coastal cell that is half sea reports about 0.5.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import h3
import h3.api.numpy_int as h3n
import numpy as np
import numpy.typing as npt

from nadir.errors import GridError
from nadir.registro import REGLAS

RESOLUCIONES = (5, 6, 7, 8)
"""Closed set of resolutions: national, regional, main analysis, case zones."""

ANALISIS = 7
"""Main analysis resolution (docs/design.md, section 2)."""

Arreglo = npt.NDArray[np.float64]


def comprobar_resolucion(resolucion: int) -> None:
    """Raise unless the resolution is one of the observatory's."""
    if resolucion not in RESOLUCIONES:
        raise GridError(f"resolution {resolucion} is not one of {RESOLUCIONES}")


def celda(latitud: float, longitud: float, resolucion: int) -> str:
    """H3 cell that contains a point."""
    comprobar_resolucion(resolucion)
    if not (-90.0 <= latitud <= 90.0 and -180.0 <= longitud <= 180.0):
        raise GridError(f"point out of range: ({latitud}, {longitud})")
    return str(h3.latlng_to_cell(latitud, longitud, resolucion))


def padre(identificador: str, resolucion: int) -> str:
    """Parent cell at a coarser observatory resolution."""
    comprobar_resolucion(resolucion)
    if not h3.is_valid_cell(identificador):
        raise GridError(f"not a valid H3 cell: {identificador!r}")
    propia = int(h3.get_resolution(identificador))
    if resolucion > propia:
        raise GridError(f"resolution {resolucion} is finer than the cell's ({propia})")
    return str(h3.cell_to_parent(identificador, resolucion))


def diametro_medio_km(resolucion: int) -> float:
    """Mean diameter of a cell in km, defined as twice the mean hexagon edge length.

    A regular hexagon's widest extent (corner to corner) is twice its edge, so this is the
    diameter of the circle that holds an average cell. Resolutions 0 to 15 are accepted, because
    rules that compare sizes may look at resolutions coarser than the observatory's.
    """
    if not 0 <= resolucion <= 15:
        raise GridError(f"H3 resolutions go from 0 to 15, not {resolucion}")
    return 2 * float(h3.average_hexagon_edge_length(resolucion, "km"))


def area_km2(identificador: str) -> float:
    """Exact area of a cell in km²."""
    if not h3.is_valid_cell(identificador):
        raise GridError(f"not a valid H3 cell: {identificador!r}")
    return float(h3.cell_area(identificador, "km^2"))


@dataclass(frozen=True)
class Agregado:
    """Value of one cell after aggregation."""

    valor: float
    fraccion_valida: float


def _por_celda(
    regla: str, grupo: npt.NDArray[np.intp], n: int, valores: Arreglo, pesos: Arreglo
) -> Arreglo:
    """One value per cell for a rule, with pixels already grouped by cell."""
    if regla in ("fraccion", "media"):
        resultado: Arreglo = np.bincount(grupo, weights=pesos * valores, minlength=n) / np.bincount(
            grupo, weights=pesos, minlength=n
        )
        return resultado
    if regla == "suma":
        return np.bincount(grupo, weights=valores, minlength=n).astype(np.float64)
    if regla == "conteo":
        return np.bincount(grupo, minlength=n).astype(np.float64)
    # moda: area-weighted mode; ties go to the smallest value so the result is deterministic.
    clases, clase = np.unique(valores, return_inverse=True)
    par = grupo.astype(np.int64) * clases.size + clase
    pares, inversa = np.unique(par, return_inverse=True)
    sumas = np.bincount(inversa, weights=pesos)
    celda_par = pares // clases.size
    clase_par = pares % clases.size
    orden = np.lexsort((clase_par, -sumas, celda_par))
    primeras = orden[np.r_[True, np.diff(celda_par[orden]) != 0]]
    moda: Arreglo = np.empty(n, dtype=np.float64)
    moda[celda_par[primeras]] = clases[clase_par[primeras]]
    return moda


def agregar(
    latitudes: Sequence[float] | Arreglo,
    longitudes: Sequence[float] | Arreglo,
    valores: Sequence[float] | Arreglo,
    areas_km2: Sequence[float] | Arreglo,
    resolucion: int,
    regla: str,
) -> dict[str, Agregado]:
    """Aggregate pixel centres to H3 cells with one of the registry's rules.

    - `valores`: one value per pixel; NaN marks no data and is excluded.
    - `areas_km2`: area of each pixel in km² (weights and valid-area fraction).
    - `fraccion`: values must be 0 or 1; result is the area-weighted share of 1.
    - `suma`, `media`, `moda`, `conteo`: sum, area-weighted mean, area-weighted mode, count of
      valid pixels.

    Cells are returned sorted by id. A cell with no valid pixel is left out. For whole rasters,
    `nadir.celdas` assigns each pixel once and tabulates every band from that map.
    """
    comprobar_resolucion(resolucion)
    if regla not in REGLAS:
        raise GridError(f"aggregation rule {regla!r} is not one of {REGLAS}")
    lat = np.asarray(latitudes, dtype=np.float64)
    lon = np.asarray(longitudes, dtype=np.float64)
    val = np.asarray(valores, dtype=np.float64)
    are = np.asarray(areas_km2, dtype=np.float64)
    if not (lat.shape == lon.shape == val.shape == are.shape) or lat.ndim != 1:
        raise GridError("latitudes, longitudes, values and areas must be 1-D and the same length")
    if np.any(~np.isfinite(are)) or np.any(are <= 0):
        raise GridError("pixel areas must be finite and positive")
    validos = ~np.isnan(val)
    if regla == "fraccion" and np.any((val[validos] != 0) & (val[validos] != 1)):
        raise GridError("rule 'fraccion' needs values 0 or 1")
    lat, lon, val, are = lat[validos], lon[validos], val[validos], are[validos]
    if np.any((np.abs(lat) > 90.0) | (np.abs(lon) > 180.0)):
        raise GridError("point out of range")
    convertir = h3n.latlng_to_cell
    enteras = np.fromiter(
        (convertir(a, b, resolucion) for a, b in zip(lat.tolist(), lon.tolist(), strict=True)),
        dtype=np.int64,
        count=lat.size,
    )
    nombres = np.asarray([h3.int_to_str(int(c)) for c in np.unique(enteras).tolist()])
    ordenados = np.argsort(nombres, kind="stable")
    rango = np.empty(nombres.size, dtype=np.intp)
    rango[ordenados] = np.arange(nombres.size)
    _, grupo_entero = np.unique(enteras, return_inverse=True)
    grupo = rango[grupo_entero]
    n = nombres.size
    valor = _por_celda(regla, grupo, n, val, are)
    area_valida = np.bincount(grupo, weights=are, minlength=n)
    resultado: dict[str, Agregado] = {}
    for posicion, ident in enumerate(nombres[ordenados].tolist()):
        valida = float(area_valida[posicion]) / area_km2(ident)
        resultado[ident] = Agregado(valor=float(valor[posicion]), fraccion_valida=min(valida, 1.0))
    return resultado
