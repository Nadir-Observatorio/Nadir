"""Class groupings of the study and loading of its area tables.

Tables are aggregated on load into eight categories that keep everything the study uses: level 1
of the legend plus codes 9 (forestry), 59 (primary forest) and 60 (secondary forest).
"""

from pathlib import Path
from typing import Any

import h3.api.numpy_int as h3n
import numpy as np
import numpy.typing as npt
import pandas as pd
import pyarrow.parquet as pq

BOSQUE, NO_BOSC, AGRO, SINVEG, AGUA = 1, 2, 3, 4, 5
SILVICULTURA, BOSQUE_PRIMARIO, BOSQUE_SECUNDARIO = 5, 2, 3
NOMBRES = {
    1: "Formación boscosa",
    2: "Formación natural no boscosa",
    3: "Agropecuaria y silvicultura",
    4: "Área sin vegetación",
    5: "Cuerpo de agua",
}
# One minus the user's accuracy of each level-1 class in the land cover accuracy assessment.
ERROR_USUARIO = {1: 1 - 0.84, 2: 1 - 0.70, 3: 1 - 0.91, 4: 1 - 0.83, 5: 1 - 0.80}


def _categorias() -> npt.NDArray[np.int8]:
    cat = np.zeros(256, dtype=np.int8)
    for codigo in (1, 3, 67):
        cat[codigo] = 1
    cat[59], cat[60] = 2, 3
    for codigo in (10, 11, 12, 29, 63, 66):
        cat[codigo] = 4
    cat[9] = 5
    for codigo in (14, 15, 18):
        cat[codigo] = 6
    for codigo in (22, 23, 24, 25, 61):
        cat[codigo] = 7
    for codigo in (26, 33, 34):
        cat[codigo] = 8
    return cat


CATEGORIA = _categorias()


def _categorias_c1() -> npt.NDArray[np.int8]:
    """Collection 1: the categories of collection 2, plus the network codes it lacks."""
    cat = CATEGORIA.copy()
    extra = {1: (4, 5, 6, 49), 4: (13, 32, 50), 6: (19, 20, 21, 35, 36, 39, 40, 41, 46, 47, 48),
             7: (30, 68), 8: (31,)}
    for categoria, codigos in extra.items():
        cat[list(codigos)] = categoria
    return cat


CATEGORIA_C1 = _categorias_c1()
CATEGORIA_A_NIVEL1 = np.array([0, 1, 1, 1, 2, 3, 3, 4, 5], dtype=np.int8)

Tabla = Any  # a pandas DataFrame; pandas ships no type information


def cargar(ruta: Path, anio_max: int | None = None) -> Tabla:
    """Read a table in batches and add it up by category, keeping every key the study uses.

    Tables built from land cover collection 1 (metadata `coleccion` = 1) use `CATEGORIA_C1`.
    """
    archivo = pq.ParquetFile(ruta)
    metadatos = archivo.schema_arrow.metadata or {}
    categoria = CATEGORIA_C1 if metadatos.get(b"coleccion") == b"1" else CATEGORIA
    nombres = archivo.schema.names
    ccols = sorted(c for c in nombres if c.startswith("c") and c[1:].isdigit())
    llaves = ["celda", "anio", "grupo", "tramo"] + (["banda"] if "banda" in nombres else []) + ccols
    partes = []
    for lote in archivo.iter_batches(batch_size=4_000_000, columns=[*llaves, "area_m2", "n_px"]):
        d = lote.to_pandas()
        if anio_max is not None:
            d = d[d.anio <= anio_max]
        for c in ccols:
            d[c] = categoria[d[c].to_numpy()]
        partes.append(d.groupby(llaves, sort=False)[["area_m2", "n_px"]].sum().reset_index())
        del d
    t = pd.concat(partes, ignore_index=True)
    return t.groupby(llaves, sort=False)[["area_m2", "n_px"]].sum().reset_index()


def padre(celdas: npt.NDArray[np.int64], resolucion: int) -> npt.NDArray[np.int64]:
    """Parent of each H3 cell at a coarser resolution."""
    unicas, inversa = np.unique(celdas, return_inverse=True)
    padres = np.array([h3n.cell_to_parent(int(c), resolucion) for c in unicas], dtype=np.int64)
    resultado: npt.NDArray[np.int64] = padres[inversa]
    return resultado


def preparar(t: Tabla) -> Tabla:
    """Add the level-1 class of each class column and the parent cells at resolutions 7 to 3."""
    t = t.copy()
    for col in [c for c in t.columns if c.startswith("c") and c[1:].isdigit()]:
        t["l" + col[1:]] = CATEGORIA_A_NIVEL1[t[col].to_numpy()]
    for res in (7, 6, 5, 4, 3):
        t[f"r{res}"] = padre(t["celda"].to_numpy(), res)
    return t
