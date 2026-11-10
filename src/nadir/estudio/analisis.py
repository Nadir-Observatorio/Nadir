"""Main analysis of the fire and land cover study, from the tables of `nadir.estudio.tablas`.

H1: among pixels that were forest formation in t - 1, burning raises the persistent conversion
to another level-1 class in t + 2 and t + 3, against the standardised never-burned control.
H3: among burned pixels whose class in t - 1 was not non-vegetated area, persistent conversion to
non-vegetated area is higher within 2 km of the urban interface than farther away. H3 is reported
as not evaluable by construction: the thematic filter of MapBiomas Fire masks burned pixels with
the land cover product itself, so no burned pixel holds non-vegetated area or water. Holm keeps
m = 2: the adjusted p of H1 is twice its own p, whatever H3 gives. Also: sensitivity analyses
without the scar database, effects by year, covariate balance, descriptive contrasts and burned
area by previous class.
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import h3.api.numpy_int as h3n
import numpy as np
import numpy.typing as npt
import pandas as pd
import pyarrow.parquet as pq

from nadir.errors import SourceError
from nadir.estudio import categorias as cat
from nadir.estudio.categorias import BOSQUE, NOMBRES, SINVEG, Tabla
from nadir.estudio.espacial import morans_i
from nadir.estudio.estimador import (
    MIN_BLOQUES,
    REPLICAS,
    SEMILLA,
    Estimador,
    bootstrap,
    h3_estimador,
    resumen,
)

ANIO_MAX_K3 = 2021
SIN_INTERVALO = "menos de 30 bloques: sin intervalo"
NOTA_BALANCE = (
    "distancia con tope de 10 km, en pasos de 40 m; estratos con control de 100 píxeles o más"
)
MOTIVO_H3 = (
    "no evaluable por construcción: el filtro temático de Fuego enmascara lo quemado con la "
    "propia Cobertura, así que ningún píxel quemado tiene Área sin vegetación ni agua"
)
NOTA_HORIZONTES = (
    "los horizontes que terminan en 2024 son los últimos disponibles: año de fuego 2021 en k3, "
    "2019 en k5 y 2022 en k2"
)
SIN_QUEMADO = "sin área quemada comparable en esta clase previa"
NOTA_2025 = "2025: enero a abril (la exportación llega hasta abril de 2025)"


def _sin_intervalo_si_faltan_bloques(r: dict[str, Any], bloques: int) -> dict[str, Any]:
    if bloques < MIN_BLOQUES:
        return {"efecto_pp": r["efecto_pp"], "bloques_con_quemado": bloques, "nota": SIN_INTERVALO}
    return r


def _cambio(t: Tabla) -> npt.NDArray[np.float64]:
    resultado: npt.NDArray[np.float64] = (
        (t.l1.to_numpy() != BOSQUE) & (t.l2.to_numpy() != BOSQUE)
    ).astype(float)
    return resultado


def analizar(tablas: Path, replicas: int = REPLICAS) -> dict[str, Any]:
    """Every figure of the main analysis, as a dictionary ready to save as JSON."""
    out: dict[str, Any] = {
        "semilla": SEMILLA,
        "replicas": replicas,
        "nota_horizontes": NOTA_HORIZONTES,
    }
    k3 = cat.preparar(cat.cargar(tablas / "k3.parquet", anio_max=ANIO_MAX_K3))
    bloques4 = np.unique(k3.r4.to_numpy())
    bloques5 = np.unique(k3.r5.to_numpy())
    bloques3 = np.unique(k3.r3.to_numpy())
    out["bloques_r4_en_dominio"] = int(bloques4.size)

    prev_b = k3.l0.to_numpy() == BOSQUE
    cambio = _cambio(k3)
    e1 = Estimador(k3, prev_b, cambio, 1, 6, 4, bloques4)
    ef, pq_, pc = e1.efecto(np.ones(e1.nb))
    rep1 = bootstrap(lambda w: e1.efecto(w)[0], e1.nb, replicas=replicas)
    h1 = _sin_intervalo_si_faltan_bloques(resumen(ef, rep1), e1.bloques_con_quemado)
    h1.update(
        {
            "p_quemado": round(100 * pq_, 3),
            "p_control_estandarizado": round(100 * pc, 3),
            "area_quemada_ha": round(e1.area_q_total / 1e4, 1),
            "area_quemada_fuera_ha": round(e1.area_q_fuera / 1e4, 1),
            "estratos": e1.estratos,
            "estratos_fuera": e1.estratos_fuera,
            "bloques_con_quemado": e1.bloques_con_quemado,
        }
    )
    q = k3[prev_b & (k3.grupo.to_numpy() == 1) & (cambio > 0)]
    dest = q.groupby("l2").area_m2.sum()
    dest = dest / dest.sum()
    h1["destino_t3"] = {NOMBRES.get(int(k), str(k)): round(100 * v, 2) for k, v in dest.items()}
    h1["error_usuario_ponderado_destino_pp"] = round(
        100 * sum(v * cat.ERROR_USUARIO.get(int(k), np.nan) for k, v in dest.items()), 2
    )

    filtro3 = (k3.grupo.to_numpy() == 1) & (k3.l0.to_numpy() != SINVEG)
    fn3, ef3, d3 = h3_estimador(k3, filtro3, 4, bloques4)
    rep3 = bootstrap(fn3, len(bloques4), replicas=replicas)
    h3: dict[str, Any] = {"evaluable": False, "motivo": MOTIVO_H3}
    h3.update(_sin_intervalo_si_faltan_bloques(resumen(ef3, rep3), d3["bloques_con_quemado"]))
    h3.update({k: (round(100 * v, 3) if k.startswith("p_") else v) for k, v in d3.items()})

    if "p_unilateral" in h1:
        h1["p_holm"] = min(1.0, 2 * h1["p_unilateral"])
        h1["apoyada"] = h1["p_holm"] < 0.05
        h1["e_estrella_pp"] = h1["ic95_pp"][0]
    h1["holm_m"] = 2
    out["H1"], out["H3"] = h1, h3
    out["moran_residuo_H1"] = morans_i(e1, ef)

    out["sensibilidades"] = _sensibilidades(
        k3, prev_b, cambio, filtro3, bloques3, bloques4, bloques5, replicas
    )
    out["por_anio"] = _por_anio(k3)
    out["balance_H1"] = _balance(tablas / "balance.parquet", e1)
    out["descriptivo"] = _descriptivo(k3, prev_b, cambio, bloques4)
    out["objetivo1_ha"] = _objetivo1(tablas / "k0.parquet")
    return out


def _primero(e: Estimador) -> Callable[[npt.NDArray[np.float64]], float]:
    def fn(w: npt.NDArray[np.float64]) -> float:
        return e.efecto(w)[0]

    return fn


def _sensibilidades(
    k3: Tabla,
    prev_b: npt.NDArray[np.bool_],
    cambio: npt.NDArray[np.float64],
    filtro3: npt.NDArray[np.bool_],
    bloques3: npt.NDArray[np.int64],
    bloques4: npt.NDArray[np.int64],
    bloques5: npt.NDArray[np.int64],
    replicas: int,
) -> dict[str, Any]:
    sens: dict[str, Any] = {}
    sin1617 = k3[~k3.anio.isin([2016, 2017])]
    pb = sin1617.l0.to_numpy() == BOSQUE
    cb = _cambio(sin1617)
    b4s = np.unique(sin1617.r4.to_numpy())
    e = Estimador(sin1617, pb, cb, 1, 6, 4, b4s)
    sens["1_sin_2016_2017_H1"] = _sin_intervalo_si_faltan_bloques(
        resumen(
            e.efecto(np.ones(e.nb))[0],
            bootstrap(lambda w: e.efecto(w)[0], e.nb, replicas=replicas),
        ),
        e.bloques_con_quemado,
    )
    f3 = (sin1617.grupo.to_numpy() == 1) & (sin1617.l0.to_numpy() != SINVEG)
    fn, ev, d = h3_estimador(sin1617, f3, 4, b4s)
    sens["1_sin_2016_2017_H3"] = _sin_intervalo_si_faltan_bloques(
        resumen(ev, bootstrap(fn, len(b4s), replicas=replicas)), d["bloques_con_quemado"]
    )
    for res, bl in ((5, bloques5), (3, bloques3)):
        eb = Estimador(k3, prev_b, cambio, 1, 6, res, bl)
        r = resumen(eb.efecto(np.ones(eb.nb))[0], bootstrap(_primero(eb), eb.nb, replicas=replicas))
        r["bloques_con_quemado"] = eb.bloques_con_quemado
        if eb.bloques_con_quemado < MIN_BLOQUES:
            r = {
                "efecto_pp": r["efecto_pp"],
                "bloques_con_quemado": eb.bloques_con_quemado,
                "nota": SIN_INTERVALO,
            }
        sens[f"4_bloques_r{res}_H1"] = r
        fnb, evb, d = h3_estimador(k3, filtro3, res, bl)
        r = resumen(evb, bootstrap(fnb, len(bl), replicas=replicas))
        if d["bloques_con_quemado"] < MIN_BLOQUES:
            r = {
                "efecto_pp": r["efecto_pp"],
                "bloques_con_quemado": d["bloques_con_quemado"],
                "nota": SIN_INTERVALO,
            }
        sens[f"4_bloques_r{res}_H3"] = r
    e7 = Estimador(k3, prev_b, cambio, 1, 7, 4, bloques4)
    sens["5_estrato_r7_H1"] = _sin_intervalo_si_faltan_bloques(
        resumen(
            e7.efecto(np.ones(e7.nb))[0],
            bootstrap(lambda w: e7.efecto(w)[0], e7.nb, replicas=replicas),
        ),
        e7.bloques_con_quemado,
    )
    return sens


def _por_anio(k3: Tabla) -> dict[int, dict[str, float | None]]:
    por_anio = {}
    for a in sorted(k3.anio.unique()):
        sub = k3[k3.anio == a]
        pbx = sub.l0.to_numpy() == BOSQUE
        e = Estimador(sub, pbx, _cambio(sub), 1, 6, 4, np.unique(sub.r4.to_numpy()))
        ef_, _, _ = e.efecto(np.ones(e.nb))
        f3a = (sub.grupo.to_numpy() == 1) & (sub.l0.to_numpy() != SINVEG)
        q3 = sub[f3a]
        y = ((q3.l1 == SINVEG) & (q3.l2 == SINVEG)).to_numpy()
        a3 = q3.area_m2.to_numpy()
        cerca = q3.banda.to_numpy() == 0
        h3a: float | None = None
        if a3[cerca].sum() > 0 and a3[~cerca].sum() > 0:
            h3a = round(
                100
                * (a3[cerca & y].sum() / a3[cerca].sum() - a3[~cerca & y].sum() / a3[~cerca].sum()),
                2,
            )
        por_anio[int(a)] = {
            "H1_efecto_pp": round(100 * ef_, 2),
            "H1_area_quemada_ha": round(e.area_q_total / 1e4, 1),
            "H3_efecto_pp": h3a,
            "H3_area_quemada_ha": round(a3.sum() / 1e4, 1),
        }
    return por_anio


def _balance(ruta: Path, e1: Estimador) -> dict[str, Any]:
    columnas = ["w", "w_alt", "w_alt2", "w_dist", "w_dist2"]
    bal = pd.read_parquet(ruta)
    bal = bal[(bal.anio <= ANIO_MAX_K3) & (bal.nivel1_previo == BOSQUE)]
    bal["r6"] = cat.padre(bal.celda.to_numpy(), 6)
    qb = bal[bal.grupo == 1]
    cbal = bal[bal.grupo == 0].groupby(["r6", "tramo", "anio"])[columnas].sum()
    qs = qb.groupby(["r6", "tramo", "anio"])[columnas].sum()
    acc = np.zeros(5)
    peso_total = 0.0
    filas_q = []
    for (c6, tr, an), row in qs.iterrows():
        s = e1.pos.get((int(c6), int(tr), int(an)))
        if s is None or not e1.elegible[s]:
            continue
        pool = np.zeros(5)
        for v in h3n.grid_disk(int(c6), 1):
            k = (int(v), tr, an)
            if k in cbal.index:
                pool += cbal.loc[k].to_numpy()
        if pool[0] <= 0:
            continue
        m = pool / pool[0]
        acc += row["w"] * m
        peso_total += row["w"]
        filas_q.append(row.to_numpy())
    mq = np.sum(filas_q, axis=0)
    mq = mq / mq[0]
    mc = acc / peso_total

    def smd(i1: int, i2: int) -> float:
        vq = mq[i2] - mq[i1] ** 2
        vc = mc[i2] - mc[i1] ** 2
        return round(float((mq[i1] - mc[i1]) / np.sqrt((vq + vc) / 2)), 3)

    return {
        "altitud_media_quemado_m": round(float(mq[1]), 1),
        "altitud_media_control_est_m": round(float(mc[1]), 1),
        "smd_altitud": smd(1, 2),
        "distancia_media_quemado_m": round(float(mq[3]), 0),
        "distancia_media_control_est_m": round(float(mc[3]), 0),
        "smd_distancia": smd(3, 4),
        "nota": NOTA_BALANCE,
    }


def _descriptivo(
    k3: Tabla,
    prev_b: npt.NDArray[np.bool_],
    cambio: npt.NDArray[np.float64],
    bloques4: npt.NDArray[np.int64],
) -> dict[str, Any]:
    desc: dict[str, Any] = {}
    c0, c1, c2 = k3.c0.to_numpy(), k3.c1.to_numpy(), k3.c2.to_numpy()
    l0, l1, l2 = k3.l0.to_numpy(), k3.l1.to_numpy(), k3.l2.to_numpy()

    def prop(
        previa: npt.NDArray[np.bool_], res: npt.NDArray[np.bool_], grupo_q: int = 1
    ) -> dict[str, Any]:
        e = Estimador(k3, previa, res.astype(float), grupo_q, 6, 4, bloques4)
        ef_, p1, p0 = e.efecto(np.ones(e.nb))
        if not np.isfinite(ef_):
            return {"nota": SIN_QUEMADO, "area_quemada_ha": round(e.area_q_total / 1e4, 1)}
        return {
            "p_quemado": round(100 * p1, 2),
            "p_control_est": round(100 * p0, 2),
            "diferencia_pp": round(100 * ef_, 2),
            "area_quemada_en_estratos_ha": round((e.area_q_total - e.area_q_fuera) / 1e4, 1),
            "area_quemada_fuera_ha": round(e.area_q_fuera / 1e4, 1),
        }

    silv, prim, sec = cat.SILVICULTURA, cat.BOSQUE_PRIMARIO, cat.BOSQUE_SECUNDARIO
    desc["6.2_retorno_silvicultura_k3"] = prop(c0 == silv, (c1 == silv) & (c2 == silv))
    desc["6.2_retorno_formacion_boscosa_k3"] = prop(prev_b, (l1 == BOSQUE) & (l2 == BOSQUE))
    desc["6.3_bosque_primario_a_secundario_k3"] = prop(c0 == prim, (c1 == sec) & (c2 == sec))
    desc["6.3_formacion_boscosa_a_silvicultura_k3"] = prop(prev_b, (c1 == silv) & (c2 == silv))
    desc["6.3_requemados_H1"] = prop(prev_b, cambio > 0, grupo_q=2)
    matriz = {}
    for pcl in range(1, 6):
        fila = {}
        for dcl in range(1, 6):
            if dcl == pcl:
                continue
            fila[NOMBRES[dcl]] = prop(l0 == pcl, (l1 == dcl) & (l2 == dcl))
        matriz[NOMBRES[pcl]] = fila
    desc["matriz_nivel1_persistente_k3"] = matriz
    return desc


def _objetivo1(ruta: Path) -> dict[Any, Any]:
    k0 = cat.cargar(ruta)
    k0["l0"] = cat.CATEGORIA_A_NIVEL1[k0.c0.to_numpy()]
    qk = k0[k0.grupo.isin([1, 2])]
    tab = qk.groupby(["anio", "l0"]).area_m2.sum().unstack(fill_value=0) / 1e4
    salida: dict[Any, Any] = {
        int(a): {NOMBRES.get(int(c), str(c)): round(float(v), 1) for c, v in row.items()}
        for a, row in tab.iterrows()
    }
    salida["nota"] = NOTA_2025
    return salida


BITS_SENSIBILIDAD = {"2": (2, 0), "3": (0, 1)}
SENSIBILIDADES_LFS = {
    "2": ("2_lfs_ignicion_2008_2012", True),
    "3": ("3_control_sin_lfs_1985_2018", False),
}


def sensibilidad_lfs(tablas: Path, numero: str, replicas: int = REPLICAS) -> dict[str, Any]:
    """H1 (and H3 for sensitivity 2) on tables built with a scar exclusion.

    Sensitivity 2 takes tables without the pixels of scars ignited in 2008-2012, in every group;
    sensitivity 3, tables whose control leaves every scar of the 1985-2018 seasons. H3 compares
    burned pixels only, so sensitivity 3 does not change it.
    """
    if numero not in SENSIBILIDADES_LFS:
        raise SourceError(f"scar sensitivities are {sorted(SENSIBILIDADES_LFS)}, not {numero}")
    etiqueta, con_h3 = SENSIBILIDADES_LFS[numero]
    esperado = BITS_SENSIBILIDAD[numero]
    ruta = tablas / "k3.parquet"
    metadatos = pq.read_schema(ruta).metadata or {}
    if b"excluido_por_cicatrices_m2" not in metadatos:
        raise SourceError(f"{ruta} was built without a scar exclusion")
    bits = (int(metadatos[b"excluir_bits"]), int(metadatos[b"excluir_bits_control"]))
    if bits != esperado:
        raise SourceError(
            f"sensitivity {numero} needs tables built with flags {esperado} (every group, control "
            f"only); {ruta} has {bits}"
        )
    fuera_m2 = float(metadatos[b"excluido_por_cicatrices_m2"])
    k3 = cat.preparar(cat.cargar(ruta, anio_max=ANIO_MAX_K3))
    bloques4 = np.unique(k3.r4.to_numpy())
    prev_b = k3.l0.to_numpy() == BOSQUE
    e = Estimador(k3, prev_b, _cambio(k3), 1, 6, 4, bloques4)
    h1 = _sin_intervalo_si_faltan_bloques(
        resumen(e.efecto(np.ones(e.nb))[0], bootstrap(_primero(e), e.nb, replicas=replicas)),
        e.bloques_con_quemado,
    )
    h1["area_quemada_ha"] = round(e.area_q_total / 1e4, 1)
    out: dict[str, Any] = {
        "semilla": SEMILLA,
        "replicas": replicas,
        "excluir_bits": int(metadatos[b"excluir_bits"]),
        "excluir_bits_control": int(metadatos[b"excluir_bits_control"]),
        "excluido_por_cicatrices_ha_sumado_por_anio": round(fuera_m2 / 1e4, 1),
        f"{etiqueta}_H1": h1,
    }
    if con_h3:
        filtro3 = (k3.grupo.to_numpy() == 1) & (k3.l0.to_numpy() != SINVEG)
        fn3, ef3, d3 = h3_estimador(k3, filtro3, 4, bloques4)
        h3: dict[str, Any] = {"evaluable": False, "motivo": MOTIVO_H3}
        h3.update(
            _sin_intervalo_si_faltan_bloques(
                resumen(ef3, bootstrap(fn3, len(bloques4), replicas=replicas)),
                d3["bloques_con_quemado"],
            )
        )
        out[f"{etiqueta}_H3"] = h3
    return out


ANIO_MAX_C1 = 2019
NOTA_SENSIBILIDAD_7 = (
    "Exploratory (amendment E-5.1): H1 with land cover collection 1 and with collection 2, both "
    "on fire years 2013-2019, same estimator, strata, blocks, replicates and seed. Each "
    "collection keeps its own valid pixels; the area left out for an invalid code is reported "
    "for each. No formal test of the difference."
)


def _h1_en_anios(ruta: Path, anio_max: int, replicas: int) -> dict[str, Any]:
    k3 = cat.preparar(cat.cargar(ruta, anio_max=anio_max))
    bloques4 = np.unique(k3.r4.to_numpy())
    e = Estimador(k3, k3.l0.to_numpy() == BOSQUE, _cambio(k3), 1, 6, 4, bloques4)
    ef, pq_, pc = e.efecto(np.ones(e.nb))
    h1 = _sin_intervalo_si_faltan_bloques(
        resumen(ef, bootstrap(_primero(e), e.nb, replicas=replicas)), e.bloques_con_quemado
    )
    metadatos = pq.read_schema(ruta).metadata or {}
    h1.update(
        {
            "p_quemado": round(100 * pq_, 3),
            "p_control_estandarizado": round(100 * pc, 3),
            "area_quemada_ha": round(e.area_q_total / 1e4, 1),
            "area_quemada_fuera_ha": round(e.area_q_fuera / 1e4, 1),
            "estratos": e.estratos,
            "bloques_con_quemado": e.bloques_con_quemado,
        }
    )
    if b"excluido_codigo_no_valido_m2" in metadatos:
        h1["excluido_codigo_no_valido_ha_sumado_por_anio"] = round(
            float(metadatos[b"excluido_codigo_no_valido_m2"]) / 1e4, 1
        )
    return h1


def sensibilidad_coleccion(
    tablas_c2: Path, tablas_c1: Path, replicas: int = REPLICAS
) -> dict[str, Any]:
    """Sensitivity 7: H1 with land cover collection 1 against collection 2, fire years 2013-2019."""
    rutas = {"coleccion_2": tablas_c2 / "k3.parquet", "coleccion_1": tablas_c1 / "k3.parquet"}
    for etiqueta, ruta in rutas.items():
        metadatos = pq.read_schema(ruta).metadata or {}
        es_c1 = metadatos.get(b"coleccion") == b"1"
        if es_c1 != (etiqueta == "coleccion_1"):
            raise SourceError(f"{ruta} is not a {etiqueta.replace('_', ' ')} table")
        if b"excluido_por_cicatrices_m2" in metadatos:
            raise SourceError(f"{ruta} was built with a scar exclusion; sensitivity 7 has none")
    out: dict[str, Any] = {
        "semilla": SEMILLA,
        "replicas": replicas,
        "anios_fuego": [2013, ANIO_MAX_C1],
        "nota": NOTA_SENSIBILIDAD_7,
    }
    for etiqueta, ruta in rutas.items():
        out[f"7_H1_{etiqueta}"] = _h1_en_anios(ruta, ANIO_MAX_C1, replicas)
    out["diferencia_c1_menos_c2_pp_descriptiva"] = round(
        out["7_H1_coleccion_1"]["efecto_pp"] - out["7_H1_coleccion_2"]["efecto_pp"], 2
    )
    return out


def escribir(resultados: dict[str, Any], salida: Path) -> None:
    """Save the results as UTF-8 JSON, refusing to overwrite."""
    if salida.exists():
        raise SourceError(f"refusing to overwrite {salida}")
    texto = json.dumps(resultados, ensure_ascii=False, indent=1, allow_nan=False) + "\n"
    salida.write_bytes(texto.encode())
