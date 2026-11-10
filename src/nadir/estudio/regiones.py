"""Results by region: burned area and the H1 effect split by administrative region.

A pixel takes the region that contains the centre of its resolution-8 H3 cell (about 0.7 km^2).
Pixels of a cell whose centre falls in no region go to `sin_region`, and the area of the cells
that cross a regional boundary is reported, so the reader can bound the error of the rule. The
regional boundaries are used only to add up areas; they are not drawn.

The effect of each region is the same split as the cell maps (`mapas.repartir`): the shares of all
regions, `sin_region` included, add up to the effect of H1, or the code stops. Regions keep their
effect whatever their burned forest area (`MIN_HA_REGION` is 0); the area is reported beside it.
"""

import json
from pathlib import Path

import h3.api.numpy_int as h3n
import numpy as np
import numpy.typing as npt
import pandas as pd

from nadir.errors import SourceError
from nadir.estudio import mapas
from nadir.estudio.categorias import BOSQUE

SIN_REGION = "sin_region"
MIN_HA_REGION = 0.0


def _poligonos(ruta: Path, campo: str) -> list[tuple[str, object]]:
    from shapely.geometry import shape

    datos = json.loads(ruta.read_text(encoding="utf-8"))
    salida = [(str(f["properties"][campo]), shape(f["geometry"])) for f in datos["features"]]
    nombres = [n for n, _ in salida]
    if len(set(nombres)) != len(nombres):
        raise SourceError(f"{ruta}: repeated names in field {campo}")
    return salida


def asignar(
    celdas: npt.NDArray[np.int64], ruta: Path, campo: str = "NAME"
) -> tuple[dict[int, str], float]:
    """Region of each resolution-8 cell by its centre, and the area of cells crossing a boundary."""
    from shapely import STRtree, points
    from shapely.geometry import Polygon

    regiones = _poligonos(ruta, campo)
    arbol = STRtree([g for _, g in regiones])
    unicas = np.unique(celdas)
    centros = np.array([h3n.cell_to_latlng(int(c)) for c in unicas])
    pts = points(centros[:, 1], centros[:, 0])
    pares = arbol.query(pts, predicate="within")
    region = np.full(len(unicas), SIN_REGION, dtype=object)
    vistos = np.zeros(len(unicas), dtype=int)
    for i, j in zip(pares[0], pares[1], strict=True):
        region[i] = regiones[j][0]
        vistos[i] += 1
    if (vistos > 1).any():
        raise SourceError("a cell centre falls in two regions; the boundaries overlap")
    polis = [Polygon([(lon, lat) for lat, lon in h3n.cell_to_boundary(int(c))]) for c in unicas]
    cruces = arbol.query(polis, predicate="intersects")[0]
    n = np.bincount(cruces, minlength=len(unicas))
    en_borde = (n > 1) | ((n == 1) & (region == SIN_REGION))
    borde = float(sum(h3n.cell_area(int(c), unit="km^2") for c in unicas[en_borde]) * 100)
    return {int(c): str(r) for c, r in zip(unicas, region, strict=True)}, borde


def por_region(
    tablas: Path, ruta: Path, campo: str = "NAME"
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Table by region and the metadata of the assignment."""
    b = mapas.base(tablas)
    mapa, borde_ha = asignar(b.k3.celda.to_numpy(), ruta, campo)
    bosque = b.k3.l0.to_numpy() == BOSQUE
    grupo1 = b.k3.grupo.to_numpy() == 1
    etiqueta = np.array([mapa[int(c)] for c in b.k3.celda.to_numpy()], dtype=object)
    t = mapas.repartir(
        b, etiqueta[grupo1], etiqueta[bosque & grupo1], MIN_HA_REGION, nombre="region"
    )
    meta: dict[str, object] = {
        "h1_pp": round(float(b.h1_pp), 4),
        "celdas_r8_en_borde_ha": round(borde_ha, 1),
        "regla": "region of the centre of the resolution-8 H3 cell",
    }
    return t.sort_values("quemado_ha", ascending=False).reset_index(drop=True), meta


def escribir(tabla: pd.DataFrame, meta: dict[str, object], carpeta: Path) -> list[Path]:
    """CSV of the table and JSON of the metadata, LF line ends. Never overwrites."""
    rutas = [carpeta / "regiones.csv", carpeta / "regiones.json"]
    for r in rutas:
        if r.exists():
            raise SourceError(f"refusing to overwrite {r}")
    carpeta.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(rutas[0], index=False, float_format="%.4f", lineterminator="\n")
    texto = json.dumps(meta, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    rutas[1].write_bytes(texto.encode("utf-8"))
    return rutas
