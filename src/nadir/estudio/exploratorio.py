"""Exploratory analyses, declared before their figures were computed and kept apart from the
confirmatory block (no Holm adjustment, no supported or not supported label).

Urbanisation after fire: among pixels whose level-1 class in t - 1 was forest formation,
non-forest natural formation or farming and forestry, the share that is Infrastructure (code 24)
in t + 4 and t + 5, burned against the standardised never-burned control. The Fire thematic
filter hides a burned pixel that is code 24 in every year t to t + 3, so those pixels leave both
groups and the figure is a lower bound of urbanisation.

The same comparison runs on the `l5` table, where the fire comes from Landscape Fire Scars
(seasons 2014-2018, previous class in s - 2, outcome in s + 4 and s + 5), which does not go through
the Fire filter; there is no mask to apply.

When no burned pixel holds code 24 in any year the table reads, the filter has removed every
burned pixel that was ever Infrastructure and the comparison is not evaluable by construction:
the output says so and carries no effect.
"""

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import pyarrow.parquet as pq

from nadir.errors import SourceError
from nadir.estudio import categorias as cat
from nadir.estudio.categorias import NOMBRES, Tabla
from nadir.estudio.estimador import REPLICAS, SEMILLA, Estimador, bootstrap, resumen

INFRAESTRUCTURA = 24
CLASES_PREVIAS = (cat.BOSQUE, cat.NO_BOSC, cat.AGRO)
MIN_BLOQUES = 30
ETIQUETAS = {
    "u5": "exploratorio: cota inferior de la urbanización (la de t a t + 3 queda oculta)",
    "l5": "exploratorio: fuego de Landscape Fire Scars, urbanización a s + 4 y s + 5",
}
MOTIVO_NO_EVALUABLE = (
    "no evaluable por construcción: ningún píxel quemado tiene código 24 en t - 1, t + 4 ni t + 5, "
    "mientras el control sí; el filtro temático de Fuego elimina todo píxel quemado que haya sido "
    "Infraestructura en algún año"
)


def _cargar(ruta: Path, con_mascara: bool) -> tuple[Tabla, float, float, float]:
    archivo = pq.ParquetFile(ruta)
    if con_mascara and "m24" not in archivo.schema.names:
        raise SourceError(f"{ruta} is not the exploratory u5 table")
    columnas = ["celda", "anio", "grupo", "tramo", "banda", "c0", "c1", "c2"]
    partes = []
    quemado_24 = control_24 = 0.0
    leer = [*columnas, *(["m24"] if con_mascara else []), "area_m2", "n_px"]
    for lote in archivo.iter_batches(batch_size=4_000_000, columns=leer):
        d = lote.to_pandas()
        if not con_mascara:
            d["m24"] = np.uint8(0)
        alguna = (d.c0 == INFRAESTRUCTURA) | (d.c1 == INFRAESTRUCTURA) | (d.c2 == INFRAESTRUCTURA)
        quemado_24 += float(d.area_m2[alguna & (d.grupo > 0)].sum())
        control_24 += float(d.area_m2[alguna & (d.grupo == 0)].sum())
        d["l0"] = cat.CATEGORIA_A_NIVEL1[cat.CATEGORIA[d.c0.to_numpy()]]
        d["y"] = ((d.c1 == INFRAESTRUCTURA) & (d.c2 == INFRAESTRUCTURA)).astype(np.int8)
        llaves = ["celda", "anio", "grupo", "tramo", "banda", "m24", "l0", "y"]
        partes.append(d.groupby(llaves, sort=False)[["area_m2", "n_px"]].sum().reset_index())
    t = pd.concat(partes, ignore_index=True)
    t = t.groupby(list(t.columns[:8]), sort=False)[["area_m2", "n_px"]].sum().reset_index()
    t = t[t.l0.isin(CLASES_PREVIAS)]
    oculto = float(t[t.m24 == 1].area_m2.sum())
    t = t[t.m24 == 0].copy()
    for res in (7, 6, 4):
        t[f"r{res}"] = cat.padre(t["celda"].to_numpy(), res)
    return t, oculto, quemado_24, control_24


def urbanizacion(tablas: Path, replicas: int = REPLICAS, tabla: str = "u5") -> dict[str, Any]:
    """Urbanisation after fire by previous class, with intervals, and near or far from the
    interface as description, from the `u5` (Fire) or `l5` (Landscape Fire Scars) table."""
    if tabla not in ETIQUETAS:
        raise SourceError(f"urbanisation reads the tables {sorted(ETIQUETAS)}, not {tabla}")
    t, oculto, quemado_24, control_24 = _cargar(tablas / f"{tabla}.parquet", tabla == "u5")
    bloques = np.unique(t.r4.to_numpy())
    y = t.y.to_numpy().astype(float)
    l0 = t.l0.to_numpy()
    out: dict[str, Any] = {
        "etiqueta": ETIQUETAS[tabla],
        "tabla": tabla,
        "semilla": SEMILLA,
        "replicas": replicas,
        "anios_o_temporadas_de_fuego": [int(t.anio.min()), int(t.anio.max())],
        "fuera_por_mascara_simetrica_ha_sumado_por_anio": round(oculto / 1e4, 1),
        "quemado_con_codigo_24_ha_sumado_por_anio": round(quemado_24 / 1e4, 1),
        "control_con_codigo_24_ha_sumado_por_anio": round(control_24 / 1e4, 1),
    }
    if quemado_24 == 0 and control_24 > 0:
        out["evaluable"] = False
        out["motivo"] = MOTIVO_NO_EVALUABLE
        return out
    out["evaluable"] = True
    grupos = {NOMBRES[c]: l0 == c for c in CLASES_PREVIAS}
    grupos["Las tres clases"] = np.isin(l0, CLASES_PREVIAS)
    por_clase: dict[str, Any] = {}
    for nombre, previa in grupos.items():
        e = Estimador(t, previa, y, 1, 6, 4, bloques)
        ef, p1, p0 = e.efecto(np.ones(e.nb))
        if not np.isfinite(ef):
            por_clase[nombre] = {"nota": "sin área quemada comparable"}
            continue

        def fn(w: npt.NDArray[np.float64], e: Estimador = e) -> float:
            return e.efecto(w)[0]

        r = resumen(ef, bootstrap(fn, e.nb, replicas=replicas))
        if e.bloques_con_quemado < MIN_BLOQUES:
            r = {"efecto_pp": r["efecto_pp"], "nota": "menos de 30 bloques: sin intervalo"}
        r.update({
            "p_quemado": round(100 * p1, 3),
            "p_control_estandarizado": round(100 * p0, 3),
            "area_quemada_en_estratos_ha": round((e.area_q_total - e.area_q_fuera) / 1e4, 1),
            "bloques_con_quemado": e.bloques_con_quemado,
        })
        por_clase[nombre] = r
    out["por_clase_previa"] = por_clase
    interfaz: dict[str, Any] = {}
    previa = grupos["Las tres clases"]
    for etiqueta, banda in (("cerca_menos_de_2_km", 0), ("lejos_2_km_o_mas", 1)):
        sub = t[t.banda.to_numpy() == banda]
        if not ((sub.grupo == 1).any() and (sub.grupo == 0).any()):
            interfaz[etiqueta] = {"nota": "sin área quemada o sin control"}
            continue
        en_banda = t.banda.to_numpy() == banda
        e = Estimador(sub, previa[en_banda], y[en_banda], 1, 6, 4, np.unique(sub.r4.to_numpy()))
        ef, p1, p0 = e.efecto(np.ones(e.nb))
        interfaz[etiqueta] = (
            {"nota": "sin área quemada comparable"}
            if not np.isfinite(ef)
            else {
                "diferencia_pp": round(100 * ef, 3),
                "p_quemado": round(100 * p1, 3),
                "p_control_estandarizado": round(100 * p0, 3),
                "area_quemada_en_estratos_ha": round((e.area_q_total - e.area_q_fuera) / 1e4, 1),
            }
        )
    out["interfaz_descriptivo"] = interfaz
    return out
