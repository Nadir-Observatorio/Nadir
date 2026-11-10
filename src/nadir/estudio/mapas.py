"""Results by H3 cell, for the maps of the study (resolutions 6 and 7).

The effect of H1 is an area-weighted mean over burned pixels of (outcome minus the standardised
control proportion of the pixel's stratum). That sum splits exactly over any partition of the
burned pixels, so each cell gets its own effect and its share of the total, and the shares add up
to the effect of H1. The control of a stratum keeps the definition of the estimator (resolution 6
cell with its first ring, slope band and fire year); a resolution-7 cell uses the strata of its
resolution-6 parent. No interval per cell: the bootstrap stays at the level of the whole domain.

Cells with less than `MIN_HA_CELDA` of burned forest in eligible strata carry their burned area
but no effect (the column is empty), because a few hectares give an unstable proportion.
Boundaries of regions and municipalities are not drawn; the cell outlines are H3 cells.
"""

import json
from pathlib import Path
from typing import Any, NamedTuple

import h3.api.numpy_int as h3n
import numpy as np
import numpy.typing as npt
import pandas as pd

from nadir.errors import SourceError
from nadir.estudio import categorias as cat
from nadir.estudio.analisis import ANIO_MAX_K3, _cambio
from nadir.estudio.categorias import BOSQUE
from nadir.estudio.estimador import Estimador

MIN_HA_CELDA = 100.0
RESOLUCIONES = (6, 7)


def _contribuciones(
    e: Estimador,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """Per burned record: area, outcome minus its stratum's control proportion, eligibility."""
    uno = np.ones(e.nb)
    c = e.C @ uno
    cc = e.Cc @ uno
    b = e.B @ uno
    ok = e.elegible & (c > 0) & (b > 0)
    pc = np.divide(cc, c, out=np.zeros_like(cc), where=c > 0)
    s = e.q_estrato
    return e.q_area, e.q_y - pc[s], ok[s]


class Base(NamedTuple):
    """The H1 estimator and the per-record terms that every split of the effect reuses."""

    k3: pd.DataFrame
    e: Estimador
    area: npt.NDArray[np.float64]
    desvio: npt.NDArray[np.float64]
    ok: npt.NDArray[np.bool_]
    h1_pp: float


def base(tablas: Path) -> Base:
    """Load k3, build the H1 estimator and the contribution of each burned forest record."""
    k3 = cat.preparar(cat.cargar(tablas / "k3.parquet", anio_max=ANIO_MAX_K3))
    bosque = k3.l0.to_numpy() == BOSQUE
    e = Estimador(k3, bosque, _cambio(k3), 1, 6, 4, np.unique(k3.r4.to_numpy()))
    area, desvio, ok = _contribuciones(e)
    return Base(k3, e, area, desvio, ok, 100 * e.efecto(np.ones(e.nb))[0])


def repartir(
    b: Base,
    clave_quemado: npt.NDArray[Any],
    clave_bosque: npt.NDArray[Any],
    min_ha: float | None = None,
    nombre: str = "celda",
) -> pd.DataFrame:
    """Burned area and H1 effect for each group of a split; stops if the shares miss H1.

    `clave_quemado` labels every once-burned record of k3, `clave_bosque` the burned forest
    records of the estimator, in the same order as `b.area`.
    """
    quemado = b.k3[b.k3.grupo.to_numpy() == 1]
    todo = quemado.area_m2.groupby(clave_quemado).sum().rename("quemado_ha") / 1e4
    d = pd.DataFrame(
        {
            nombre: clave_bosque,
            "a": b.area,
            "a_ok": np.where(b.ok, b.area, 0.0),
            "ay": np.where(b.ok, b.area * b.e.q_y, 0.0),
            "ad": np.where(b.ok, b.area * b.desvio, 0.0),
        }
    )
    g = d.groupby(nombre).sum()
    total_ok = float(b.area[b.ok].sum())
    t = pd.DataFrame(index=todo.index.union(g.index))
    t.index.name = nombre
    t["quemado_ha"] = todo.reindex(t.index).fillna(0.0)
    t["bosque_quemado_ha"] = (g.a / 1e4).reindex(t.index).fillna(0.0)
    t["bosque_comparable_ha"] = (g.a_ok / 1e4).reindex(t.index).fillna(0.0)
    con = t.bosque_comparable_ha >= (MIN_HA_CELDA if min_ha is None else min_ha)
    efecto = (g.ad / g.a_ok).reindex(t.index)
    p_q = (g.ay / g.a_ok).reindex(t.index)
    t["efecto_pp"] = (100 * efecto).where(con)
    t["p_quemado_pct"] = (100 * p_q).where(con)
    t["p_control_pct"] = (100 * (p_q - efecto)).where(con)
    t["aporte_pp"] = (100 * g.ad / total_ok).reindex(t.index).fillna(0.0)
    suma = float(t.aporte_pp.sum())
    if abs(suma - b.h1_pp) > 1e-6:
        raise SourceError(f"{nombre}: shares add up to {suma}, not {b.h1_pp}")
    return t.reset_index()


def por_celda(tablas: Path) -> dict[int, pd.DataFrame]:
    """One table per resolution with burned area and the H1 effect of each cell.

    Stops if the shares of a resolution do not add up to the effect of H1.
    """
    b = base(tablas)
    bosque = b.k3.l0.to_numpy() == BOSQUE
    q = b.k3[bosque & (b.k3.grupo.to_numpy() == 1)]
    quemado = b.k3[b.k3.grupo.to_numpy() == 1]
    return {
        res: repartir(b, quemado[f"r{res}"].to_numpy(), q[f"r{res}"].to_numpy())
        for res in RESOLUCIONES
    }


def _poligono(celda: int) -> list[list[list[float]]]:
    anillo = [[round(lon, 6), round(lat, 6)] for lat, lon in h3n.cell_to_boundary(celda)]
    return [[*anillo, anillo[0]]]


def _valor(v: object) -> object:
    if isinstance(v, float | np.floating):
        return None if not np.isfinite(v) else round(float(v), 4)
    return v


def escribir(celdas: dict[int, pd.DataFrame], carpeta: Path) -> list[Path]:
    """GeoJSON and CSV of each resolution, with LF line ends on every system. Never overwrites."""
    rutas = []
    for res in celdas:
        for ext in ("geojson", "csv"):
            ruta = carpeta / f"celdas_r{res}.{ext}"
            if ruta.exists():
                raise SourceError(f"refusing to overwrite {ruta}")
            rutas.append(ruta)
    carpeta.mkdir(parents=True, exist_ok=True)
    for res, t in celdas.items():
        tabla = t.copy()
        tabla.insert(0, "h3", [h3n.int_to_str(int(c)) for c in tabla.celda])
        tabla = tabla.drop(columns="celda")
        tabla.to_csv(
            carpeta / f"celdas_r{res}.csv", index=False, float_format="%.4f", lineterminator="\n"
        )
        rasgos = [
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": _poligono(int(c))},
                "properties": {k: _valor(v) for k, v in fila.items() if k != "celda"}
                | {"h3": h3n.int_to_str(int(c))},
            }
            for c, fila in zip(t.celda, t.to_dict(orient="records"), strict=True)
        ]
        coleccion = {"type": "FeatureCollection", "features": rasgos}
        texto = json.dumps(coleccion, ensure_ascii=False, allow_nan=False)
        (carpeta / f"celdas_r{res}.geojson").write_bytes((texto + "\n").encode("utf-8"))
    return rutas
