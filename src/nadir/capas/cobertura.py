"""MapBiomas Chile land cover, collection 2.0, read from its Earth Engine export in the bank.

MapBiomas Chile serves its maps only through Google Earth Engine (the bucket links of the
collection page answer 403, and the platform downloads statistics, not maps). Nadir exports the
asset published as collection 2.0 with `gee/exportar_mapbiomas_7_regiones.js`, in its native grid
(EPSG:4326, about 30 m), and keeps the zip from Google Drive in the bank as received.

The export covers the seven regions mapped by MapBiomas Fire Chile, collection 1 (Valparaíso to
La Araucanía), years 2012 to 2024: 2012 is the year before the first fire year. Values: the
legend codes of collection 2; 0 is MapBiomas' own "no data"; 255 is outside the seven regions.
"""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from nadir import banco, exportacion
from nadir.errors import SourceError

ACTIVO = (
    "projects/mapbiomas-chile/assets/LULC/COLLECTION-02/CLASSIFICATIONS/"
    "classification-final/clasificacion-final-2"
)
PAGINA = "https://chile.mapbiomas.org/mapas-de-la-coleccion/"
PREFIJO = "mbchile_cobertura_c2_final2_2012_2024"
ANIOS_ESTUDIO = tuple(range(2012, 2025))
CARPETA = Path("mapbiomas-chile") / "cobertura-c2"
SIN_DATO = 0
FUERA = 255
FRANJA = 1024
"""Rows read at a time: one strip of a full-width band is about 16 MB."""

FUENTE = banco.Fuente(
    organismo="MapBiomas Chile",
    conjunto="Cobertura y Uso del Suelo, mapas anuales, Colección 2.0 (1999-2024)",
    fuente_pagina=PAGINA,
    licencia="MapBiomas-Chile",
    atribucion=(
        "MapBiomas Chile Project - Collection 2.0 of the Annual Series of Land Cover and Land "
        "Use Maps of Chile, consulted on [date] via the link: https://chile.mapbiomas.org"
    ),
)

NIVEL_1: dict[int, str] = {
    1: "formacion_boscosa",
    3: "formacion_boscosa",
    59: "formacion_boscosa",
    60: "formacion_boscosa",
    67: "formacion_boscosa",
    10: "formacion_natural_no_boscosa",
    11: "formacion_natural_no_boscosa",
    12: "formacion_natural_no_boscosa",
    29: "formacion_natural_no_boscosa",
    63: "formacion_natural_no_boscosa",
    66: "formacion_natural_no_boscosa",
    9: "agropecuaria_y_silvicultura",
    14: "agropecuaria_y_silvicultura",
    15: "agropecuaria_y_silvicultura",
    18: "agropecuaria_y_silvicultura",
    22: "area_sin_vegetacion",
    23: "area_sin_vegetacion",
    24: "area_sin_vegetacion",
    25: "area_sin_vegetacion",
    61: "area_sin_vegetacion",
    26: "cuerpo_de_agua",
    33: "cuerpo_de_agua",
    34: "cuerpo_de_agua",
    27: "no_observado",
}
"""Legend codes of collection 2 and their level-1 class. 27 (not observed) is kept apart."""


def banda(anio: int) -> str:
    """Band name of one year in the asset and in the export."""
    if anio not in ANIOS_ESTUDIO:
        raise SourceError(f"the export covers {ANIOS_ESTUDIO[0]}-{ANIOS_ESTUDIO[-1]}, not {anio}")
    return f"classification_{anio}"


def abrir(bruto: Path) -> exportacion.Mosaico:
    """Open the export from the bank, after checking every recorded file and the band list."""
    carpeta = bruto / CARPETA
    mosaico = exportacion.abrir(banco.zips(carpeta), PREFIJO)
    esperadas = tuple(banda(a) for a in ANIOS_ESTUDIO)
    if mosaico.bandas != esperadas:
        mosaico.cerrar()
        raise SourceError(f"{PREFIJO}: bands {mosaico.bandas}, expected {esperadas}")
    if mosaico.sin_dato != FUERA:
        mosaico.cerrar()
        raise SourceError(f"{PREFIJO}: no-data value {mosaico.sin_dato}, expected {FUERA}")
    return mosaico


@dataclass(frozen=True)
class Informe:
    """What one export holds: its grid and the area of every code, by year, in hectares."""

    grilla: exportacion.Grilla
    hectareas: dict[int, dict[int, float]]
    pixeles: dict[int, Counter[int]]


def describir(mosaico: exportacion.Mosaico, anios: tuple[int, ...] = ANIOS_ESTUDIO) -> Informe:
    """Area and pixel count of every code in each year, and a refusal of any unknown code."""
    validos = set(NIVEL_1) | {SIN_DATO, FUERA}
    hectareas: dict[int, dict[int, float]] = {}
    pixeles: dict[int, Counter[int]] = {}
    for anio in anios:
        area = np.zeros(256, dtype=np.float64)
        cuenta = np.zeros(256, dtype=np.int64)
        for fila, datos in mosaico.franjas(banda(anio), FRANJA):
            filas = datos.shape[0]
            pesos = np.repeat(mosaico.grilla.area_filas_m2(fila, filas), datos.shape[1])
            plano = datos.ravel()
            area += np.bincount(plano, weights=pesos, minlength=256)
            cuenta += np.bincount(plano, minlength=256)
        presentes = {int(c) for c in np.flatnonzero(cuenta)}
        extranos = sorted(presentes - validos)
        if extranos:
            raise SourceError(f"{anio}: codes outside the collection 2 legend: {extranos}")
        hectareas[anio] = {c: float(area[c]) / 10_000 for c in sorted(presentes)}
        pixeles[anio] = Counter({c: int(cuenta[c]) for c in sorted(presentes)})
    return Informe(mosaico.grilla, hectareas, pixeles)
