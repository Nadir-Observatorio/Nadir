"""Standardised effect of burning with a spatial block bootstrap.

Every estimate is an area-weighted proportion. A stratum is an H3 cell (resolution 6 by default)
with its first ring, crossed with the slope band and the fire year, because the control is
measured in the same years as the burned area. The control of a stratum is the never-burned area
of the same previous class and slope band in the seven cells of the disk. A stratum enters when
its control holds at least `MIN_CONTROL_PX` pixels.

Bootstrap: blocks are H3 cells (resolution 4 by default); every block of the domain is resampled
with replacement, with all its burned and control content; percentile 95 % intervals; one-sided
p = share of replicates with an effect at or below zero.
"""

from collections.abc import Callable
from typing import Any

import h3.api.numpy_int as h3n
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import sparse

from nadir.estudio.categorias import SINVEG, Tabla

SEMILLA = 20240202
REPLICAS = 2000
MIN_CONTROL_PX = 100
MIN_BLOQUES = 30

Arreglo = npt.NDArray[np.float64]


class Estimador:
    """Standardised effect of one previous class, ready for the block bootstrap."""

    def __init__(
        self,
        t: Tabla,
        previa: npt.NDArray[np.bool_],
        resultado: Arreglo,
        grupo_q: int,
        res_estrato: int,
        res_bloque: int,
        bloques_universo: npt.NDArray[np.int64],
    ) -> None:
        q = t[previa & (t.grupo.to_numpy() == grupo_q)]
        rq = resultado[previa & (t.grupo.to_numpy() == grupo_q)]
        c = t[previa & (t.grupo.to_numpy() == 0)]
        rc = resultado[previa & (t.grupo.to_numpy() == 0)]
        bidx = {int(b): i for i, b in enumerate(bloques_universo)}
        self.nb = len(bloques_universo)
        ce = f"r{res_estrato}"
        cb = f"r{res_bloque}"
        sk = pd.DataFrame(
            {"cel": q[ce].to_numpy(), "tr": q.tramo.to_numpy(), "an": q.anio.to_numpy()}
        )
        est, sinv = np.unique(sk.to_records(index=False), return_inverse=True)
        self.ns = len(est)
        qb = np.array([bidx[int(b)] for b in q[cb].to_numpy()])
        self.B = sparse.csr_matrix((q.area_m2.to_numpy(), (sinv, qb)), shape=(self.ns, self.nb))
        self.Bc = sparse.csr_matrix(
            (q.area_m2.to_numpy() * rq, (sinv, qb)), shape=(self.ns, self.nb)
        )
        pos = {(int(r[0]), int(r[1]), int(r[2])): i for i, r in enumerate(est)}
        self.pos = pos
        self.q_estrato = sinv
        self.q_r7 = q["r7"].to_numpy()
        self.q_area = q.area_m2.to_numpy()
        self.q_y = rq
        cg = (
            pd.DataFrame(
                {
                    "cel": c[ce].to_numpy(),
                    "tr": c.tramo.to_numpy(),
                    "an": c.anio.to_numpy(),
                    "bl": [bidx[int(x)] for x in c[cb].to_numpy()],
                    "a": c.area_m2.to_numpy(),
                    "ac": c.area_m2.to_numpy() * rc,
                    "n": c.n_px.to_numpy(),
                }
            )
            .groupby(["cel", "tr", "an", "bl"], sort=False)
            .sum()
            .reset_index()
        )
        filas: list[int] = []
        colsb: list[int] = []
        a: list[float] = []
        ac: list[float] = []
        npx: list[float] = []
        vecinos = {int(u): [int(v) for v in h3n.grid_disk(int(u), 1)] for u in cg.cel.unique()}
        for cel, tr, an, bl, ar_, ac_, n_ in cg.itertuples(index=False, name=None):
            for v in vecinos[int(cel)]:
                s = pos.get((v, int(tr), int(an)))
                if s is not None:
                    filas.append(s)
                    colsb.append(int(bl))
                    a.append(ar_)
                    ac.append(ac_)
                    npx.append(n_)
        self.C = sparse.csr_matrix((a, (filas, colsb)), shape=(self.ns, self.nb))
        self.Cc = sparse.csr_matrix((ac, (filas, colsb)), shape=(self.ns, self.nb))
        cnt = np.bincount(
            np.asarray(filas, dtype=int), weights=np.asarray(npx, dtype=float), minlength=self.ns
        )
        self.elegible = cnt >= MIN_CONTROL_PX
        self.area_q_total = float(self.B.sum())
        self.area_q_fuera = float(np.asarray(self.B.sum(axis=1)).ravel()[~self.elegible].sum())
        self.estratos = int(self.ns)
        self.estratos_fuera = int((~self.elegible).sum())
        bq = np.asarray(self.B[self.elegible].sum(axis=0)).ravel()
        self.bloques_con_quemado = int((bq > 0).sum())

    def efecto(self, w: Arreglo) -> tuple[float, float, float]:
        """Effect, burned proportion and standardised control proportion for block weights."""
        b = self.B @ w
        bc = self.Bc @ w
        c = self.C @ w
        cc = self.Cc @ w
        ok = self.elegible & (c > 0) & (b > 0)
        if not ok.any():
            return np.nan, np.nan, np.nan
        pq_ = bc[ok].sum() / b[ok].sum()
        pc = (b[ok] * (cc[ok] / c[ok])).sum() / b[ok].sum()
        return pq_ - pc, pq_, pc


def bootstrap(
    fn: Callable[[Arreglo], float], nb: int, semilla: int = SEMILLA, replicas: int = REPLICAS
) -> Arreglo:
    """Value of `fn` for each replicate of multinomial block weights."""
    rng = np.random.default_rng(semilla)
    out = np.empty(replicas)
    for r in range(replicas):
        w = np.bincount(rng.integers(0, nb, nb), minlength=nb).astype(float)
        out[r] = fn(w)
    return out


def resumen(est: float, rep: Arreglo) -> dict[str, Any]:
    """Effect in percentage points, percentile interval and one-sided p."""
    rep = rep[np.isfinite(rep)]
    lo, hi = np.percentile(rep, [2.5, 97.5])
    p = float((rep <= 0).mean())
    salida: dict[str, Any] = {
        "efecto_pp": round(100 * est, 3),
        "ic95_pp": [round(100 * lo, 3), round(100 * hi, 3)],
        "p_unilateral": p,
        "replicas_validas": int(rep.size),
    }
    if p == 0:
        salida["p_cota"] = f"< 1/{rep.size}"
    return salida


def holm(p: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjustment of one-sided p values."""
    orden = sorted(p, key=lambda k: p[k])
    m = len(p)
    out: dict[str, float] = {}
    prev = 0.0
    for i, k in enumerate(orden):
        v = min(1.0, max(prev, (m - i) * p[k]))
        out[k] = v
        prev = v
    return out


def h3_estimador(
    t: Tabla, filtro: npt.NDArray[np.bool_], res_bloque: int, bloques: npt.NDArray[np.int64]
) -> tuple[Callable[[Arreglo], float], float, dict[str, Any]]:
    """Difference in persistent conversion to non-vegetated area, near minus far from the
    urban interface, among burned pixels selected by `filtro`."""
    q = t[filtro]
    y = ((q.l1.to_numpy() == SINVEG) & (q.l2.to_numpy() == SINVEG)).astype(float)
    bidx = {int(b): i for i, b in enumerate(bloques)}
    bb = np.array([bidx[int(b)] for b in q[f"r{res_bloque}"].to_numpy()])
    a = q.area_m2.to_numpy()
    cerca = q.banda.to_numpy() == 0
    nb = len(bloques)
    a0 = np.bincount(bb[cerca], weights=a[cerca], minlength=nb)
    y0 = np.bincount(bb[cerca], weights=(a * y)[cerca], minlength=nb)
    a1 = np.bincount(bb[~cerca], weights=a[~cerca], minlength=nb)
    y1 = np.bincount(bb[~cerca], weights=(a * y)[~cerca], minlength=nb)

    def fn(w: Arreglo) -> float:
        return float((y0 @ w) / (a0 @ w) - (y1 @ w) / (a1 @ w))

    base = np.ones(nb)
    return (
        fn,
        fn(base),
        {
            "p_cerca": float(y0.sum() / a0.sum()),
            "p_lejos": float(y1.sum() / a1.sum()),
            "area_cerca_ha": round(a0.sum() / 1e4, 1),
            "area_lejos_ha": round(a1.sum() / 1e4, 1),
            "bloques_con_quemado": int(((a0 + a1) > 0).sum()),
        },
    )
