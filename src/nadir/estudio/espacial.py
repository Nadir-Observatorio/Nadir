"""Spatial autocorrelation of the residual effect."""

from typing import Any

import h3.api.numpy_int as h3n
import numpy as np
import numpy.typing as npt
import pandas as pd

from nadir.estudio.estimador import SEMILLA, Estimador


def morans_i(est: Estimador, efecto_dominio: float, permutaciones: int = 999) -> dict[str, Any]:
    """Moran's I of the residual effect by resolution-7 cell, first-ring neighbours.

    The residual of a cell is its burned proportion minus its standardised control proportion,
    minus the domain effect. The p value is one-sided, from random permutations of the residuals.
    """
    w = np.ones(est.nb)
    c, cc = est.C @ w, est.Cc @ w
    pc = np.where(c > 0, cc / np.where(c > 0, c, 1.0), np.nan)
    ok = est.elegible[est.q_estrato] & np.isfinite(pc[est.q_estrato])
    df = (
        pd.DataFrame(
            {
                "r7": est.q_r7[ok],
                "a": est.q_area[ok],
                "ay": est.q_area[ok] * est.q_y[ok],
                "apc": est.q_area[ok] * pc[est.q_estrato[ok]],
            }
        )
        .groupby("r7")
        .sum()
    )
    efecto = (df.ay - df.apc) / df.a
    z = (efecto - efecto_dominio).to_numpy()
    z = z - z.mean()
    celdas = df.index.to_numpy()
    pos = {int(c): i for i, c in enumerate(celdas)}
    pares_i: list[int] = []
    pares_j: list[int] = []
    for i, celda in enumerate(celdas):
        for v in h3n.grid_ring(int(celda), 1):
            j = pos.get(int(v))
            if j is not None:
                pares_i.append(i)
                pares_j.append(j)
    pi, pj = np.array(pares_i), np.array(pares_j)
    n, total = len(z), len(pi)

    def moran(x: npt.NDArray[np.float64]) -> float:
        return (n / total) * float((x[pi] * x[pj]).sum()) / float((x * x).sum())

    obs = moran(z)
    rng = np.random.default_rng(SEMILLA)
    perm = np.array([moran(rng.permutation(z)) for _ in range(permutaciones)])
    return {
        "I": round(obs, 4),
        "p_permutacion_unilateral": round(
            float((1 + (perm >= obs).sum()) / (permutaciones + 1)), 4
        ),
        "celdas_r7": int(n),
        "pares_vecinos": int(total),
    }
