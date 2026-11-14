import json

import h3.api.numpy_int as h3n
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from nadir.__main__ import main
from nadir.errors import SourceError
from nadir.estudio import categorias, estimador, interfaz
from test_estudio_analisis import _escribir


def _dos_lados():
    """One stratum per side. Load category 1 is forest formation, 4 non-forest.

    Inside: burned 100 of 400 changes, control 50 of 200: effect 0.25 - 0.25 = 0.
    Outside: burned 300 of 400 changes, control 20 of 200: effect 0.75 - 0.10 = 0.65.
    """
    celda = int(h3n.latlng_to_cell(-35.0, -72.0, 7))
    filas = []
    for banda, burned, control in (
        (0, [(100.0, True), (300.0, False)], [(50.0, True), (150.0, False)]),
        (1, [(300.0, True), (100.0, False)], [(20.0, True), (180.0, False)]),
    ):
        for grupo, lista in ((1, burned), (0, control)):
            for area, cambia in lista:
                filas.append({"celda": celda, "anio": 2015, "grupo": grupo, "tramo": 0,
                              "banda": banda, "c0": 1, "c1": 4 if cambia else 1,
                              "c2": 4 if cambia else 1, "area_m2": area, "n_px": 60})
    return categorias.preparar(pd.DataFrame(filas))


def test_joint_bootstrap_repeats_the_draws_of_the_single_one():
    fns = [lambda w: float(w[0]), lambda w: float(w[1] - w[2])]
    junto = estimador.bootstrap_conjunto(fns, 5, replicas=40)
    for j, fn in enumerate(fns):
        assert (junto[:, j] == estimador.bootstrap(fn, 5, replicas=40)).all()


def test_two_sided_summary():
    r = estimador.resumen_bilateral(0.1, np.array([-0.1, 0.0, 0.1, 0.2, 0.3, np.nan]))
    assert r["efecto_pp"] == 10.0
    assert r["p_bilateral"] == pytest.approx(0.8)  # 2 x min(2/5, 4/5)
    assert r["replicas_validas"] == 5
    assert estimador.resumen_bilateral(-0.1, np.array([-0.2, -0.1]))["p_cota"] == "< 2/2"
    assert estimador.resumen_bilateral(0.0, np.array([0.0, 0.0]))["p_bilateral"] == 1.0


def test_each_side_keeps_its_own_control_and_d_is_inside_minus_outside():
    t = _dos_lados()
    bloques = np.unique(t.r4.to_numpy())
    r = interfaz.contraste(t, bloques, replicas=5)
    assert r["dentro"]["efecto_pp"] == pytest.approx(0.0)
    assert r["fuera"]["efecto_pp"] == pytest.approx(65.0)
    assert r["dentro"]["p_control_estandarizado"] == pytest.approx(25.0)
    assert r["fuera"]["p_quemado"] == pytest.approx(75.0)
    assert r["D_dentro_menos_fuera"]["efecto_pp"] == pytest.approx(-65.0)
    assert "nota" in r["D_dentro_menos_fuera"]  # one block: no interval
    assert r["dentro"]["area_quemada_ha"] == 0.0  # 400 m², rounded to 0.1 ha
    assert r["dentro"]["estratos"] == 1 and r["fuera"]["bloques_con_quemado"] == 1


def test_a_table_without_band_is_refused():
    t = _dos_lados().drop(columns="banda")
    with pytest.raises(SourceError, match="no interface band"):
        interfaz.contraste(t, np.unique(t.r4.to_numpy()))


def test_holm_of_the_interface_module():
    assert interfaz.holm_v3(0.03, 1, None) == 0.03
    assert interfaz.holm_v3(0.01, 2, 0.04) == pytest.approx(0.02)
    assert interfaz.holm_v3(0.04, 2, 0.01) == pytest.approx(0.04)
    assert interfaz.holm_v3(0.03, 2, 0.02) == pytest.approx(0.04)
    assert interfaz.holm_v3(0.6, 2, 0.7) == 1.0
    with pytest.raises(SourceError):
        interfaz.holm_v3(0.01, 1, 0.02)
    with pytest.raises(SourceError):
        interfaz.holm_v3(0.01, 2, None)
    with pytest.raises(SourceError):
        interfaz.holm_v3(0.01, 3, 0.02)


def _con_banda_invertida(origen, destino):
    destino.mkdir()
    t = pq.read_table(origen / "k3.parquet")
    banda = pa.array(1 - t.column("banda").to_numpy(), type=pa.uint8())
    t = t.set_column(t.schema.get_field_index("banda"), "banda", banda)
    pq.write_table(t, destino / "k3.parquet")
    return destino


def test_full_h4_is_deterministic_and_the_wui_tables_are_read(tmp_path):
    carpeta = _escribir(tmp_path / "t")
    a = interfaz.analizar_h4(carpeta, 1, replicas=30)
    b = interfaz.analizar_h4(carpeta, 1, replicas=30)
    assert json.dumps(a) == json.dumps(b)
    json.dumps(a, allow_nan=False)
    assert list(a) == ["semilla", "replicas", "holm_m", "bloques_r4_en_dominio", "H4"]
    d = a["H4"]["D_dentro_menos_fuera"]
    assert d["efecto_pp"] == pytest.approx(
        a["H4"]["dentro"]["efecto_pp"] - a["H4"]["fuera"]["efecto_pp"], abs=2e-3
    )
    if "p_bilateral" in d:
        assert d["p_holm"] == d["p_bilateral"]
    invertida = _con_banda_invertida(carpeta, tmp_path / "wui")
    c = interfaz.analizar_h4(carpeta, 1, tablas_wui=invertida, replicas=30)
    s = c["H4_sensibilidad_wui"]
    assert s["dentro"] == a["H4"]["fuera"] and s["fuera"] == a["H4"]["dentro"]
    assert s["D_dentro_menos_fuera"]["efecto_pp"] == pytest.approx(-d["efecto_pp"])


def test_holm_with_h3prima_and_mismatched_wui_tables(tmp_path):
    carpeta = _escribir(tmp_path / "t")
    a = interfaz.analizar_h4(carpeta, 2, p_h3prima=0.0, replicas=30)
    d = a["H4"]["D_dentro_menos_fuera"]
    if "p_bilateral" in d:
        assert d["p_holm"] == d["p_bilateral"]  # the other p is smaller: H4 keeps its own
    otra = _escribir(tmp_path / "otra", semilla=7)
    with pytest.raises(SourceError, match="same blocks"):
        interfaz.analizar_h4(carpeta, 1, tablas_wui=otra, replicas=5)


def test_command_line_writes_h4(tmp_path, capsys):
    carpeta = _escribir(tmp_path / "t")
    salida = tmp_path / "h4.json"
    args = ["estudio", "interfaz", "--tablas", str(carpeta), "--holm-m", "1",
            "--salida", str(salida), "--replicas", "10"]
    assert main(args) == 0
    assert "H4: D" in capsys.readouterr().out
    assert json.loads(salida.read_text(encoding="utf-8"))["holm_m"] == 1
    assert main(args) == 1
    assert "overwrite" in capsys.readouterr().err
    sin_m = ["estudio", "interfaz", "--tablas", str(carpeta), "--salida", str(tmp_path / "x")]
    with pytest.raises(SystemExit):
        main(sin_m)
