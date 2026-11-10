import json

import h3.api.numpy_int as h3n
import numpy as np
import pandas as pd
import pytest

from nadir.__main__ import main
from nadir.errors import SourceError
from nadir.estudio import analisis, categorias, mapas
from nadir.estudio.categorias import BOSQUE
from nadir.estudio.estimador import Estimador
from test_estudio_analisis import _escribir


def _manual(carpeta, res):
    """Cell effects recomputed from the stratum proportions, without the module."""
    k3 = categorias.preparar(categorias.cargar(carpeta / "k3.parquet", anio_max=2021))
    bosque = k3.l0.to_numpy() == BOSQUE
    e = Estimador(k3, bosque, analisis._cambio(k3), 1, 6, 4, np.unique(k3.r4.to_numpy()))
    uno = np.ones(e.nb)
    c, cc, b = e.C @ uno, e.Cc @ uno, e.B @ uno
    ok_s = e.elegible & (c > 0) & (b > 0)
    q = k3[bosque & (k3.grupo.to_numpy() == 1)]
    filas = {}
    for i, celda in enumerate(q[f"r{res}"].to_numpy()):
        s = e.q_estrato[i]
        if not ok_s[s]:
            continue
        a, y = e.q_area[i], e.q_y[i]
        acc = filas.setdefault(int(celda), [0.0, 0.0])
        acc[0] += a
        acc[1] += a * (y - cc[s] / c[s])
    return {k: (a / 1e4, 100 * d / a) for k, (a, d) in filas.items()}, 100 * e.efecto(uno)[0]


@pytest.mark.parametrize("res", [6, 7])
def test_cell_effects_match_a_direct_count_and_add_up_to_h1(tmp_path, res, monkeypatch):
    carpeta = _escribir(tmp_path / "t")
    monkeypatch.setattr(mapas, "MIN_HA_CELDA", 0.0)
    t = mapas.por_celda(carpeta)[res].set_index("celda")
    esperado, h1 = _manual(carpeta, res)
    assert t.aporte_pp.sum() == pytest.approx(h1, abs=1e-9)
    con = t[t.bosque_comparable_ha > 0]
    assert set(con.index) == set(esperado)
    for celda, (area, efecto) in esperado.items():
        assert con.loc[celda, "bosque_comparable_ha"] == pytest.approx(area)
        assert con.loc[celda, "efecto_pp"] == pytest.approx(efecto)
    diff = con.p_quemado_pct - con.p_control_pct
    assert np.allclose(diff, con.efecto_pp)


def test_small_cells_keep_their_area_but_no_effect(tmp_path):
    carpeta = _escribir(tmp_path / "t")
    assert mapas.MIN_HA_CELDA == 100.0  # declared in docs/estudio.md, section 8
    t = mapas.por_celda(carpeta)[6]
    chicas = t.bosque_comparable_ha < 100.0
    assert chicas.any()
    assert t.loc[chicas, "efecto_pp"].isna().all()
    assert t.loc[~chicas, "efecto_pp"].notna().all()
    assert (t.quemado_ha >= t.bosque_quemado_ha - 1e-9).all()


def test_shares_that_do_not_add_up_stop(tmp_path, monkeypatch):
    carpeta = _escribir(tmp_path / "t")
    original = mapas._contribuciones

    def torcida(e):
        a, d, ok = original(e)
        return a, d + 0.01, ok

    monkeypatch.setattr(mapas, "_contribuciones", torcida)
    with pytest.raises(SourceError, match="add up"):
        mapas.por_celda(carpeta)


def test_files_and_command_line(tmp_path, capsys):
    carpeta = _escribir(tmp_path / "t")
    salida = tmp_path / "mapas"
    assert main(["estudio", "mapas", "--tablas", str(carpeta), "--salida", str(salida)]) == 0
    assert "resolution 6:" in capsys.readouterr().out
    geo = json.loads((salida / "celdas_r6.geojson").read_text(encoding="utf-8"))
    csv = pd.read_csv(salida / "celdas_r6.csv")
    assert len(geo["features"]) == len(csv)
    rasgo = geo["features"][0]
    anillo = rasgo["geometry"]["coordinates"][0]
    assert anillo[0] == anillo[-1] and len(anillo) == 7
    assert h3n.get_resolution(h3n.str_to_int(rasgo["properties"]["h3"])) == 6
    with pytest.raises(SourceError, match="overwrite"):
        mapas.escribir(mapas.por_celda(carpeta), salida)
