"""Land cover trajectories of burned area, with the standardised control beside them.

For each horizon k (2, 3 and 5; k = 2 is descriptive) and each level-1 class in t - 1, the share
of burned area that holds each level-1 class persistently in t + k - 1 and t + k, and the same
share for the never-burned control standardised to the burned strata (the estimator of
`nadir.estudio.estimador`, without bootstrap). Area that is not persistent in any class is the
remainder. The transition matrix by fire year is the same for k = 3, one year at a time.
"""

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from nadir.errors import SourceError
from nadir.estudio import categorias as cat
from nadir.estudio.categorias import NOMBRES, Tabla
from nadir.estudio.estimador import Estimador

HORIZONTES = {"k2": 2, "k3": 3, "k5": 5}
ANIO_MAX = {"k2": 2022, "k3": 2021, "k5": 2019}
CLASES = tuple(range(1, 6))
NO_PERSISTENTE = "No persistente"


def _celda(t: Tabla, previa: int, bloques: npt.NDArray[np.int64]) -> dict[str, Any] | None:
    l0, l1, l2 = t.l0.to_numpy(), t.l1.to_numpy(), t.l2.to_numpy()
    mascara = l0 == previa
    if not (mascara & (t.grupo.to_numpy() == 1)).any():
        return None
    fila: dict[str, Any] = {}
    quemado_total = control_total = 0.0
    area = None
    for destino in CLASES:
        resultado = ((l1 == destino) & (l2 == destino)).astype(float)
        e = Estimador(t, mascara, resultado, 1, 6, 4, bloques)
        _, p1, p0 = e.efecto(np.ones(e.nb))
        if area is None:
            area = round((e.area_q_total - e.area_q_fuera) / 1e4, 1)
        if not np.isfinite(p1):
            return None
        fila[NOMBRES[destino]] = {
            "quemado_pct": round(100 * p1, 2),
            "control_pct": round(100 * p0, 2),
        }
        quemado_total += p1
        control_total += p0
    fila[NO_PERSISTENTE] = {
        "quemado_pct": round(100 * (1 - quemado_total), 2),
        "control_pct": round(100 * (1 - control_total), 2),
    }
    return {"area_quemada_en_estratos_ha": area, "destino": fila}


def _matriz(t: Tabla) -> dict[str, Any]:
    bloques = np.unique(t.r4.to_numpy())
    salida: dict[str, Any] = {}
    for previa in CLASES:
        celda = _celda(t, previa, bloques)
        if celda is not None:
            salida[NOMBRES[previa]] = celda
    return salida


def trayectorias(tablas: Path) -> dict[str, Any]:
    """Trajectories at k = 2, 3 and 5 and the k = 3 transition matrix by fire year."""
    out: dict[str, Any] = {
        "nota": "porcentajes del área en estratos con control de 100 píxeles o más"
    }
    k3 = None
    for nombre, k in HORIZONTES.items():
        t = cat.preparar(cat.cargar(tablas / f"{nombre}.parquet", anio_max=ANIO_MAX[nombre]))
        out[f"trayectorias_k{k}"] = _matriz(t)
        if nombre == "k3":
            k3 = t
        else:
            del t
    if k3 is None:
        raise SourceError("the k3 table is required")
    out["matriz_k3_por_anio"] = {
        int(a): _matriz(k3[k3.anio == a]) for a in sorted(k3.anio.unique())
    }
    return out
