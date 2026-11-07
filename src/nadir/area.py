"""Area of pixels on a geographic (latitude-longitude) grid, on the WGS84 ellipsoid.

On a grid in degrees a pixel has no fixed area: at 33° S a pixel of 0.000269494585° (the
MapBiomas grid) covers about 754 m², not 900 m², and less towards the south. Every area in
Nadir comes from here, never from a pixel count times a nominal size.

The area between two parallels and two meridians is exact on the ellipsoid (closed form through
the authalic latitude), so summing pixels does not drift with the size of the region.
"""

import math

import numpy as np
import numpy.typing as npt

from nadir.errors import GridError

SEMIEJE_MAYOR = 6_378_137.0
"""WGS84 semi-major axis, metres."""

ACHATAMIENTO = 1 / 298.257223563
"""WGS84 flattening."""

_E2 = ACHATAMIENTO * (2 - ACHATAMIENTO)
_E = math.sqrt(_E2)
_B = SEMIEJE_MAYOR * (1 - ACHATAMIENTO)

Arreglo = npt.NDArray[np.float64]


def _q(latitud_grados: Arreglo) -> Arreglo:
    s = np.sin(np.radians(latitud_grados))
    resultado: Arreglo = s / (1 - _E2 * s * s) + np.log((1 + _E * s) / (1 - _E * s)) / (2 * _E)
    return resultado


def _comprobar_latitudes(latitudes: Arreglo) -> None:
    if np.any(~np.isfinite(latitudes)) or np.any(np.abs(latitudes) > 90.0):
        raise GridError("latitudes must be finite and within -90 to 90 degrees")


def area_franja_m2(latitud_1: float, latitud_2: float, ancho_grados: float) -> float:
    """Exact area in m² between two parallels and two meridians `ancho_grados` apart."""
    if not math.isfinite(ancho_grados) or not 0 < ancho_grados <= 360:
        raise GridError(f"longitude width must be in (0, 360] degrees, not {ancho_grados}")
    bordes = np.asarray([latitud_1, latitud_2], dtype=np.float64)
    _comprobar_latitudes(bordes)
    q = _q(bordes)
    return float(abs(_B * _B * math.radians(ancho_grados) / 2 * (q[1] - q[0])))


def area_filas_m2(norte: float, paso: float, filas: int) -> Arreglo:
    """Area in m² of one pixel of each row of a north-up grid with square cells in degrees.

    `norte` is the latitude of the top edge, `paso` the cell size in degrees (positive) and
    `filas` the number of rows. Every pixel of a row has the same area.
    """
    if not math.isfinite(paso) or paso <= 0:
        raise GridError(f"cell size must be positive, not {paso}")
    if filas < 1:
        raise GridError(f"a grid needs at least one row, not {filas}")
    bordes = norte - paso * np.arange(filas + 1, dtype=np.float64)
    _comprobar_latitudes(bordes)
    q = _q(bordes)
    resultado: Arreglo = np.abs(_B * _B * math.radians(paso) / 2 * np.diff(q))
    return resultado


RADIO_MEDIO_KM = 6371.0088
"""Mean Earth radius (IUGG), km: the sphere H3 also uses for distances."""


def distancia_km(
    latitud_1: npt.ArrayLike,
    longitud_1: npt.ArrayLike,
    latitud_2: npt.ArrayLike,
    longitud_2: npt.ArrayLike,
) -> Arreglo:
    """Great-circle distance in km on the mean sphere (haversine), element by element.

    On the sphere the error against the ellipsoid stays under 0.5 %, enough for sizes of fire
    scars and cells; areas, which must add up exactly, come from the ellipsoid instead.
    """
    lat1, lon1, lat2, lon2 = (
        np.radians(np.asarray(v, dtype=np.float64))
        for v in (latitud_1, longitud_1, latitud_2, longitud_2)
    )
    for latitudes in (lat1, lat2):
        _comprobar_latitudes(np.degrees(latitudes))
    seno = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    resultado: Arreglo = 2 * RADIO_MEDIO_KM * np.arcsin(np.sqrt(np.clip(seno, 0.0, 1.0)))
    return resultado
