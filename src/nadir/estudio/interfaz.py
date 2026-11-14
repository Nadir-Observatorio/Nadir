"""H4 of the interface module: the effect of burning on forest loss inside and outside the
urban interface.

H4: the effect of H1 (forest formation in t - 1, persistent conversion to another level-1 class
in t + 2 and t + 3, standardised never-burned control) differs inside the interface from outside
it. The estimator, strata, minimum control and horizon are those of H1; each side keeps the
burned and the control area of its own band. Interface: the 2 km band of land cover code 24 in
t - 1 (the band of H3); sensitivity: the WUI layer of Zenodo (amendment V3-E1), from tables built
with that band.

Contrast D = effect inside - effect outside, two-sided; both effects are computed with the same
block weights in every replicate (H3 resolution-4 blocks of the whole domain, 2,000 replicates,
seed 20240202). Holm over H4 and H3' with m = 2, or m = 1 when H3' is not feasible; m is given
by the caller, declared before H4 is computed.
"""

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from nadir.errors import SourceError
from nadir.estudio import categorias as cat
from nadir.estudio.categorias import BOSQUE, Tabla
from nadir.estudio.estimador import (
    MIN_BLOQUES,
    REPLICAS,
    SEMILLA,
    Estimador,
    bootstrap_conjunto,
    resumen,
    resumen_bilateral,
)

ANIO_MAX_K3 = 2021
DENTRO = 0
SIN_INTERVALO = "menos de 30 bloques con quemado en algún lado: sin intervalo"
NOTA_LADOS = (
    "cada lado se estandariza con su propio control; el efecto de H1 no es el promedio "
    "ponderado de los dos lados"
)


def _lado(e: Estimador, efecto: float, pq_: float, pc: float, rep: npt.NDArray[np.float64]
          ) -> dict[str, Any]:
    r: dict[str, Any] = (
        resumen(efecto, rep)
        if e.bloques_con_quemado >= MIN_BLOQUES
        else {"efecto_pp": round(100 * efecto, 3), "nota": SIN_INTERVALO}
    )
    r.update(
        {
            "p_quemado": round(100 * pq_, 3),
            "p_control_estandarizado": round(100 * pc, 3),
            "area_quemada_ha": round(e.area_q_total / 1e4, 1),
            "area_quemada_fuera_ha": round(e.area_q_fuera / 1e4, 1),
            "estratos": e.estratos,
            "estratos_fuera": e.estratos_fuera,
            "bloques_con_quemado": e.bloques_con_quemado,
        }
    )
    return r


def contraste(t: Tabla, bloques: npt.NDArray[np.int64], replicas: int = REPLICAS
              ) -> dict[str, Any]:
    """Effect inside, effect outside and their difference, for one prepared k3 table."""
    if "banda" not in t.columns:
        raise SourceError("the table has no interface band")
    previa = t.l0.to_numpy() == BOSQUE
    cambio = ((t.l1.to_numpy() != BOSQUE) & (t.l2.to_numpy() != BOSQUE)).astype(float)
    dentro = t.banda.to_numpy() == DENTRO
    e_in = Estimador(t, previa & dentro, cambio, 1, 6, 4, bloques)
    e_out = Estimador(t, previa & ~dentro, cambio, 1, 6, 4, bloques)
    unos = np.ones(len(bloques))
    ef_in, pq_in, pc_in = e_in.efecto(unos)
    ef_out, pq_out, pc_out = e_out.efecto(unos)
    rep = bootstrap_conjunto(
        [lambda w: e_in.efecto(w)[0], lambda w: e_out.efecto(w)[0]],
        len(bloques),
        replicas=replicas,
    )
    d = ef_in - ef_out
    bloques_min = min(e_in.bloques_con_quemado, e_out.bloques_con_quemado)
    dif: dict[str, Any] = (
        resumen_bilateral(d, rep[:, 0] - rep[:, 1])
        if bloques_min >= MIN_BLOQUES
        else {"efecto_pp": round(100 * d, 3), "nota": SIN_INTERVALO}
    )
    return {
        "dentro": _lado(e_in, ef_in, pq_in, pc_in, rep[:, 0]),
        "fuera": _lado(e_out, ef_out, pq_out, pc_out, rep[:, 1]),
        "D_dentro_menos_fuera": dif,
        "nota": NOTA_LADOS,
    }


def holm_v3(p_h4: float, holm_m: int, p_h3prima: float | None) -> float:
    """Holm-adjusted p of H4 in the family of the interface module."""
    if holm_m == 1:
        if p_h3prima is not None:
            raise SourceError("with m = 1 H3' is not feasible and has no p")
        return p_h4
    if holm_m != 2 or p_h3prima is None:
        raise SourceError("Holm of the interface module needs m = 1, or m = 2 with the p of H3'")
    menor, mayor = sorted((p_h4, p_h3prima))
    ajustado = {menor: min(1.0, 2 * menor)}
    ajustado[mayor] = min(1.0, max(ajustado[menor], mayor))
    return ajustado[p_h4]


def analizar_h4(
    tablas: Path,
    holm_m: int,
    p_h3prima: float | None = None,
    tablas_wui: Path | None = None,
    replicas: int = REPLICAS,
) -> dict[str, Any]:
    """H4 with its Holm adjustment, and the WUI sensitivity when its tables are given."""
    k3 = cat.preparar(cat.cargar(tablas / "k3.parquet", anio_max=ANIO_MAX_K3))
    bloques4 = np.unique(k3.r4.to_numpy())
    out: dict[str, Any] = {
        "semilla": SEMILLA,
        "replicas": replicas,
        "holm_m": holm_m,
        "bloques_r4_en_dominio": int(bloques4.size),
    }
    h4 = contraste(k3, bloques4, replicas)
    dif = h4["D_dentro_menos_fuera"]
    if "p_bilateral" in dif:
        dif["p_holm"] = holm_v3(dif["p_bilateral"], holm_m, p_h3prima)
        dif["apoyada"] = dif["p_holm"] < 0.05
    out["H4"] = h4
    if tablas_wui is not None:
        kw = cat.preparar(cat.cargar(tablas_wui / "k3.parquet", anio_max=ANIO_MAX_K3))
        if not np.array_equal(np.unique(kw.r4.to_numpy()), bloques4):
            raise SourceError("the WUI tables do not cover the same blocks as the main tables")
        out["H4_sensibilidad_wui"] = contraste(kw, bloques4, replicas)
    return out
