"""MapBiomas Chile land cover, collection 1.0, read from its Earth Engine export in the bank.

Used only for sensitivity 7 of the study: the main effect recomputed with collection 1 in the
fire years 2013 to 2019 (collection 1 ends in 2022, the year t + 3 of fire year 2019).

The export (`gee/exportar_extensiones.js`) covers the same seven regions as the collection 2
export, years 2012 to 2022, in the grid of collection 2; the script refuses to export otherwise,
and `abrir` checks the grid again here. Values: legend codes; 0 is MapBiomas' own "no data"; 255
is outside the seven regions.

Level-1 classes follow the codes of the MapBiomas network legend, which every country shares and
which collection 2 of Chile also uses (`cobertura.NIVEL_1`). Any code outside this table stops
the reading: it is reported, never guessed.
"""

from collections import Counter
from pathlib import Path

import numpy as np

from nadir import banco, exportacion
from nadir.capas import cobertura
from nadir.errors import GridError, SourceError

PREFIJO = "mbchile_cobertura_c1_2012_2022"
ANIOS_EXPORTADOS = tuple(range(2012, 2023))
ANIOS_FUEGO = range(2013, 2020)
"""Fire years whose classes t - 1 to t + 3 all fall inside collection 1."""
CARPETA = Path("mapbiomas-chile") / "cobertura-c1"
PAGINA = "https://chile.mapbiomas.org/mapas-de-la-coleccion/"

FUENTE = banco.Fuente(
    organismo="MapBiomas Chile",
    conjunto="Cobertura y Uso del Suelo, mapas anuales, Colección 1.0 (2000-2022)",
    fuente_pagina=PAGINA,
    licencia="MapBiomas-Chile",
    atribucion=(
        "MapBiomas Chile Project - Collection 1.0 of the Annual Series of Land Cover and Land "
        "Use Maps of Chile, consulted on [date] via the link: https://chile.mapbiomas.org"
    ),
)

_FAMILIAS: dict[str, tuple[int, ...]] = {
    "formacion_boscosa": (1, 3, 4, 5, 6, 49, 59, 60, 67),
    "formacion_natural_no_boscosa": (10, 11, 12, 13, 29, 32, 50, 63, 66),
    "agropecuaria_y_silvicultura": (9, 14, 15, 18, 19, 20, 21, 35, 36, 39, 40, 41, 46, 47, 48),
    "area_sin_vegetacion": (22, 23, 24, 25, 30, 61, 68),
    "cuerpo_de_agua": (26, 31, 33, 34),
    "no_observado": (27,),
}
NIVEL_1: dict[int, str] = {c: n for n, cs in _FAMILIAS.items() for c in cs}
"""Network legend codes and their level-1 class. 27 (not observed) is kept apart."""


def _concuerda_con_coleccion_2() -> None:
    distintos = {
        c: (n, NIVEL_1.get(c)) for c, n in cobertura.NIVEL_1.items() if NIVEL_1.get(c) != n
    }
    if distintos:
        raise SourceError(f"level-1 classes differ from collection 2 for codes {distintos}")


_concuerda_con_coleccion_2()


def banda(anio: int) -> str:
    """Band name of one year in the asset and in the export."""
    if anio not in ANIOS_EXPORTADOS:
        raise SourceError(
            f"the export covers {ANIOS_EXPORTADOS[0]}-{ANIOS_EXPORTADOS[-1]}, not {anio}"
        )
    return f"classification_{anio}"


def abrir(bruto: Path) -> exportacion.Mosaico:
    """Open the export, check its bands and no-data value, and require the grid of collection 2."""
    mosaico = exportacion.abrir(banco.zips(bruto / CARPETA), PREFIJO)
    try:
        esperadas = tuple(banda(a) for a in ANIOS_EXPORTADOS)
        if mosaico.bandas != esperadas:
            raise SourceError(f"{PREFIJO}: bands {mosaico.bandas}, expected {esperadas}")
        if mosaico.sin_dato != cobertura.FUERA:
            raise SourceError(
                f"{PREFIJO}: no-data value {mosaico.sin_dato}, expected {cobertura.FUERA}"
            )
        with cobertura.abrir(bruto) as c2:
            if not exportacion.misma_grilla(mosaico.grilla, c2.grilla):
                raise GridError("collection 1 is not on the grid of collection 2; not resampled")
    except Exception:
        mosaico.cerrar()
        raise
    return mosaico


def describir(
    mosaico: exportacion.Mosaico, anios: tuple[int, ...] = ANIOS_EXPORTADOS
) -> cobertura.Informe:
    """Area and pixel count of every code in each year, and a refusal of any unknown code."""
    validos = set(NIVEL_1) | {cobertura.SIN_DATO, cobertura.FUERA}
    hectareas: dict[int, dict[int, float]] = {}
    pixeles: dict[int, Counter[int]] = {}
    for anio in anios:
        area = np.zeros(256, dtype=np.float64)
        cuenta = np.zeros(256, dtype=np.int64)
        for fila, datos in mosaico.franjas(banda(anio), cobertura.FRANJA):
            pesos = np.repeat(mosaico.grilla.area_filas_m2(fila, datos.shape[0]), datos.shape[1])
            plano = datos.ravel()
            area += np.bincount(plano, weights=pesos, minlength=256)
            cuenta += np.bincount(plano, minlength=256)
        presentes = {int(c) for c in np.flatnonzero(cuenta)}
        extranos = sorted(presentes - validos)
        if extranos:
            raise SourceError(
                f"{anio}: codes outside the level-1 table of collection 1: {extranos}"
            )
        hectareas[anio] = {c: float(area[c]) / 10_000 for c in sorted(presentes)}
        pixeles[anio] = Counter({c: int(cuenta[c]) for c in sorted(presentes)})
    return cobertura.Informe(mosaico.grilla, hectareas, pixeles)
