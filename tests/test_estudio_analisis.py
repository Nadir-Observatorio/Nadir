import json

import h3.api.numpy_int as h3n
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from nadir.__main__ import main
from nadir.errors import SourceError
from nadir.estudio import analisis, categorias, espacial, estimador


def _celdas(n, semilla=0):
    rng = np.random.default_rng(semilla)
    lat = rng.uniform(-38.0, -33.0, n)
    lon = rng.uniform(-73.0, -70.5, n)
    return np.array([h3n.latlng_to_cell(a, b, 7) for a, b in zip(lat, lon, strict=True)],
                    dtype=np.int64)


def _tabla_k3(semilla=1, filas=6000):
    rng = np.random.default_rng(semilla)
    celdas = _celdas(400, semilla)
    celda = rng.choice(celdas, filas)
    grupo = rng.choice([0, 0, 0, 1, 1, 2], filas)
    codigos = np.array([3, 59, 60, 12, 9, 15, 24, 33])
    c0 = rng.choice(codigos[:5], filas)
    quemado = grupo > 0
    c1 = np.where(quemado & (rng.random(filas) < 0.4), rng.choice(codigos, filas), c0)
    c2 = np.where(quemado & (rng.random(filas) < 0.3), rng.choice(codigos, filas), c1)
    return pa.table({
        "anio": rng.integers(2013, 2022, filas).astype(np.int16),
        "area_m2": rng.uniform(1e4, 5e5, filas),
        "banda": rng.integers(0, 2, filas).astype(np.uint8),
        "c0": c0.astype(np.uint8),
        "c1": c1.astype(np.uint8),
        "c2": c2.astype(np.uint8),
        "celda": celda,
        "grupo": grupo.astype(np.uint8),
        "n_px": rng.integers(50, 400, filas).astype(np.int64),
        "tramo": rng.integers(0, 3, filas).astype(np.uint8),
    })


def _escribir(carpeta, semilla=1):
    carpeta.mkdir()
    k3 = _tabla_k3(semilla)
    pq.write_table(k3, carpeta / "k3.parquet")
    d = k3.to_pandas()
    rng = np.random.default_rng(semilla + 10)
    bal = pd.DataFrame({
        "celda": d.celda, "anio": d.anio, "grupo": d.grupo, "tramo": d.tramo,
        "nivel1_previo": categorias.CATEGORIA_A_NIVEL1[categorias.CATEGORIA[d.c0.to_numpy()]]
        .astype(np.uint8),
        "w": d.area_m2,
    })
    alt = rng.uniform(0, 2000, len(d))
    dist = rng.uniform(0, 10000, len(d))
    bal["w_alt"] = bal.w * alt
    bal["w_alt2"] = bal.w * alt * alt
    bal["w_dist"] = bal.w * dist
    bal["w_dist2"] = bal.w * dist * dist
    pq.write_table(pa.Table.from_pandas(bal, preserve_index=False), carpeta / "balance.parquet")
    k0 = d[["anio", "area_m2", "c0", "celda", "grupo", "n_px", "tramo"]]
    pq.write_table(pa.Table.from_pandas(k0, preserve_index=False), carpeta / "k0.parquet")
    return carpeta


def _una_celda(previa, burned, control):
    """One stratum: burned and control rows in the same resolution-7 cell, year and slope.

    Classes are load categories (`categorias.CATEGORIA`): 1 is forest formation, 4 non-forest.
    """
    celda = int(h3n.latlng_to_cell(-35.0, -72.0, 7))
    filas = []
    for grupo, lista in ((1, burned), (0, control)):
        for area, cambia, n in lista:
            filas.append({"celda": celda, "anio": 2015, "grupo": grupo, "tramo": 0, "banda": 0,
                          "c0": previa, "c1": 4 if cambia else previa,
                          "c2": 4 if cambia else previa, "area_m2": area, "n_px": n})
    t = categorias.preparar(pd.DataFrame(filas))
    return t


def test_effect_is_burned_minus_standardised_control():
    t = _una_celda(1, [(100.0, True, 10), (300.0, False, 30)],
                   [(50.0, True, 60), (150.0, False, 60)])
    previa = t.l0.to_numpy() == categorias.BOSQUE
    cambio = ((t.l1.to_numpy() != 1) & (t.l2.to_numpy() != 1)).astype(float)
    bloques = np.unique(t.r4.to_numpy())
    e = estimador.Estimador(t, previa, cambio, 1, 6, 4, bloques)
    ef, p1, p0 = e.efecto(np.ones(e.nb))
    assert p1 == pytest.approx(0.25)
    assert p0 == pytest.approx(0.25)
    assert ef == pytest.approx(0.0)
    assert e.estratos == 1 and e.estratos_fuera == 0


def test_a_control_of_exactly_100_pixels_enters():
    t = _una_celda(1, [(100.0, True, 10)], [(50.0, True, 40), (150.0, False, 60)])
    previa = t.l0.to_numpy() == categorias.BOSQUE
    cambio = ((t.l1.to_numpy() != 1) & (t.l2.to_numpy() != 1)).astype(float)
    e = estimador.Estimador(t, previa, cambio, 1, 6, 4, np.unique(t.r4.to_numpy()))
    assert e.estratos_fuera == 0


def test_control_is_standardised_by_the_burned_area_of_each_stratum():
    celda = int(h3n.latlng_to_cell(-35.0, -72.0, 7))
    filas = []
    for anio, quemado, (c_si, c_no) in ((2015, 900.0, (10.0, 90.0)), (2016, 100.0, (90.0, 10.0))):
        filas.append({"celda": celda, "anio": anio, "grupo": 1, "tramo": 0, "banda": 0, "c0": 1,
                      "c1": 1, "c2": 1, "area_m2": quemado, "n_px": 10})
        for area, cambia in ((c_si, True), (c_no * 10, False)):
            filas.append({"celda": celda, "anio": anio, "grupo": 0, "tramo": 0, "banda": 0,
                          "c0": 1, "c1": 4 if cambia else 1, "c2": 4 if cambia else 1,
                          "area_m2": area, "n_px": 200})
    t = categorias.preparar(pd.DataFrame(filas))
    previa = t.l0.to_numpy() == 1
    cambio = ((t.l1.to_numpy() != 1) & (t.l2.to_numpy() != 1)).astype(float)
    e = estimador.Estimador(t, previa, cambio, 1, 6, 4, np.unique(t.r4.to_numpy()))
    _, _, pc = e.efecto(np.ones(e.nb))
    p15, p16 = 10.0 / 910.0, 90.0 / 190.0
    assert pc == pytest.approx(0.9 * p15 + 0.1 * p16)


def test_control_comes_from_the_whole_disk():
    centro = int(h3n.latlng_to_cell(-35.0, -72.0, 6))
    vecina = next(int(v) for v in h3n.grid_ring(centro, 1))
    quemada = int(h3n.cell_to_center_child(centro, 7))
    control = int(h3n.cell_to_center_child(vecina, 7))
    filas = [
        {"celda": quemada, "anio": 2015, "grupo": 1, "tramo": 0, "banda": 0, "c0": 1, "c1": 4,
         "c2": 4, "area_m2": 100.0, "n_px": 10},
        {"celda": control, "anio": 2015, "grupo": 0, "tramo": 0, "banda": 0, "c0": 1, "c1": 1,
         "c2": 1, "area_m2": 100.0, "n_px": 500},
    ]
    t = categorias.preparar(pd.DataFrame(filas))
    previa = t.l0.to_numpy() == 1
    cambio = ((t.l1.to_numpy() != 1) & (t.l2.to_numpy() != 1)).astype(float)
    e = estimador.Estimador(t, previa, cambio, 1, 6, 4, np.unique(t.r4.to_numpy()))
    assert e.estratos_fuera == 0
    assert e.efecto(np.ones(e.nb))[0] == pytest.approx(1.0)


def test_load_categories():
    cat = categorias.CATEGORIA
    assert (cat[3], cat[59], cat[60], cat[9], cat[15], cat[24], cat[33]) == (1, 2, 3, 5, 6, 7, 8)
    assert categorias.CATEGORIA_A_NIVEL1[[1, 2, 3, 4, 5, 6, 7, 8]].tolist() == [
        1, 1, 1, 2, 3, 3, 4, 5]


def test_strata_with_a_small_control_are_left_out():
    t = _una_celda(1, [(100.0, True, 10)], [(50.0, True, 40), (150.0, False, 59)])
    previa = t.l0.to_numpy() == categorias.BOSQUE
    cambio = ((t.l1.to_numpy() != 1) & (t.l2.to_numpy() != 1)).astype(float)
    e = estimador.Estimador(t, previa, cambio, 1, 6, 4, np.unique(t.r4.to_numpy()))
    assert e.estratos_fuera == 1
    assert e.area_q_fuera == pytest.approx(100.0)
    assert all(np.isnan(x) for x in e.efecto(np.ones(e.nb)))


def test_bootstrap_is_reproducible_and_weights_add_up():
    vistos = []
    a = estimador.bootstrap(lambda w: vistos.append(w.sum()) or float(w[0]), 7, replicas=50)
    b = estimador.bootstrap(lambda w: float(w[0]), 7, replicas=50)
    assert (a == b).all()
    assert set(vistos) == {7.0}
    c = estimador.bootstrap(lambda w: float(w[0]), 7, semilla=1, replicas=50)
    assert not (a == c).all()


def test_summary_and_holm():
    r = estimador.resumen(0.1, np.array([-0.1, 0.0, 0.1, 0.2, np.nan]))
    assert r["efecto_pp"] == 10.0
    assert r["p_unilateral"] == 0.5  # replicates at zero count against the effect
    assert "p_cota" not in r
    assert estimador.resumen(0.1, np.array([0.1, 0.2]))["p_cota"] == "< 1/2"
    assert r["replicas_validas"] == 4
    assert estimador.holm({"a": 0.01, "b": 0.04}) == {"a": 0.02, "b": 0.04}
    assert estimador.holm({"a": 0.03, "b": 0.02}) == {"b": 0.04, "a": 0.04}
    assert estimador.holm({"a": 0.6, "b": 0.7}) == {"a": 1.0, "b": 1.0}


def test_interface_contrast_is_near_minus_far():
    t = _una_celda(1, [(100.0, True, 10), (300.0, False, 30)], [])
    t.loc[:, "c1"] = [7, 1]  # category 7 is non-vegetated area
    t.loc[:, "c2"] = [7, 1]
    t.loc[:, "banda"] = [0, 1]
    t = categorias.preparar(t.drop(columns=[c for c in t.columns if c[0] in "lr"]))
    filtro = t.grupo.to_numpy() == 1
    fn, ef, d = estimador.h3_estimador(t, filtro, 4, np.unique(t.r4.to_numpy()))
    assert ef == pytest.approx(1.0)
    assert d["p_cerca"] == 1.0 and d["p_lejos"] == 0.0
    assert fn(np.array([2.0])) == pytest.approx(1.0)


def test_moran_detects_a_clustered_residual():
    centro = int(h3n.latlng_to_cell(-35.0, -72.0, 7))
    celdas = [int(c) for c in h3n.grid_disk(centro, 6)]
    lat0 = h3n.cell_to_latlng(centro)[1]
    filas = []
    for c in celdas:
        este = h3n.cell_to_latlng(c)[1] > lat0
        filas.append({"celda": c, "anio": 2015, "grupo": 1, "tramo": 0, "banda": 0, "c0": 1,
                      "c1": 4 if este else 1, "c2": 4 if este else 1, "area_m2": 100.0,
                      "n_px": 200})
        filas.append({"celda": c, "anio": 2015, "grupo": 0, "tramo": 0, "banda": 0, "c0": 1,
                      "c1": 1, "c2": 1, "area_m2": 100.0, "n_px": 200})
    t = categorias.preparar(pd.DataFrame(filas))
    previa = t.l0.to_numpy() == 1
    cambio = ((t.l1.to_numpy() != 1) & (t.l2.to_numpy() != 1)).astype(float)
    e = estimador.Estimador(t, previa, cambio, 1, 7, 4, np.unique(t.r4.to_numpy()))
    ef = e.efecto(np.ones(e.nb))[0]
    m = espacial.morans_i(e, ef, permutaciones=99)
    assert m["I"] > 0.5
    assert m["p_permutacion_unilateral"] == 0.01
    assert m["celdas_r7"] == len(celdas)
    conjunto = set(celdas)
    pares = sum(int(v) in conjunto for c in celdas for v in h3n.grid_ring(c, 1))
    assert m["pares_vecinos"] == pares


def test_full_analysis_is_deterministic(tmp_path):
    carpeta = _escribir(tmp_path / "t")
    a = analisis.analizar(carpeta, replicas=20)
    b = analisis.analizar(carpeta, replicas=20)
    assert json.dumps(a, ensure_ascii=False) == json.dumps(b, ensure_ascii=False)
    json.dumps(a, allow_nan=False)  # strict JSON: no NaN anywhere
    assert list(a) == ["semilla", "replicas", "nota_horizontes", "bloques_r4_en_dominio", "H1",
                       "H3", "moran_residuo_H1", "sensibilidades", "por_anio", "balance_H1",
                       "descriptivo", "objetivo1_ha"]
    assert a["H3"]["evaluable"] is False and "no evaluable" in a["H3"]["motivo"]
    assert "apoyada" not in a["H3"] and "p_holm" not in a["H3"]
    assert a["H1"]["holm_m"] == 2
    if "p_unilateral" in a["H1"]:
        assert a["H1"]["p_holm"] == min(1.0, 2 * a["H1"]["p_unilateral"])
    assert a["objetivo1_ha"]["nota"].startswith("2025")
    assert set(a["sensibilidades"]) == {
        "1_sin_2016_2017_H1", "1_sin_2016_2017_H3", "4_bloques_r5_H1", "4_bloques_r5_H3",
        "4_bloques_r3_H1", "4_bloques_r3_H3", "5_estrato_r7_H1",
    }
    salida = tmp_path / "r.json"
    analisis.escribir(a, salida)
    assert json.loads(salida.read_text(encoding="utf-8"))["semilla"] == estimador.SEMILLA
    with pytest.raises(SourceError, match="overwrite"):
        analisis.escribir(a, salida)


def test_command_line_writes_the_results(tmp_path, capsys):
    carpeta = _escribir(tmp_path / "t")
    salida = tmp_path / "r.json"
    args = ["estudio", "analizar", "--tablas", str(carpeta), "--salida", str(salida),
            "--replicas", "10"]
    assert main(args) == 0
    assert "H1: effect" in capsys.readouterr().out
    assert main(args) == 1
    assert "overwrite" in capsys.readouterr().err


def test_scar_sensitivities(tmp_path):
    carpeta = _escribir(tmp_path / "t")
    with pytest.raises(SourceError, match="without a scar exclusion"):
        analisis.sensibilidad_lfs(carpeta, "2", replicas=5)
    k3 = pq.read_table(carpeta / "k3.parquet")
    k3 = k3.replace_schema_metadata({"excluir_bits": "2", "excluir_bits_control": "0",
                                     "excluido_por_cicatrices_m2": "123456.0"})
    pq.write_table(k3, carpeta / "k3.parquet")
    dos = analisis.sensibilidad_lfs(carpeta, "2", replicas=5)
    assert set(dos) == {"semilla", "replicas", "excluir_bits", "excluir_bits_control",
                        "excluido_por_cicatrices_ha_sumado_por_anio", "2_lfs_ignicion_2008_2012_H1",
                        "2_lfs_ignicion_2008_2012_H3"}
    assert dos["excluido_por_cicatrices_ha_sumado_por_anio"] == 12.3
    with pytest.raises(SourceError, match="needs tables built with flags"):
        analisis.sensibilidad_lfs(carpeta, "3", replicas=5)
    assert dos["2_lfs_ignicion_2008_2012_H3"]["evaluable"] is False
    k3 = k3.replace_schema_metadata({"excluir_bits": "0", "excluir_bits_control": "1",
                                     "excluido_por_cicatrices_m2": "5.0"})
    pq.write_table(k3, carpeta / "k3.parquet")
    tres = analisis.sensibilidad_lfs(carpeta, "3", replicas=5)
    assert "3_control_sin_lfs_1985_2018_H1" in tres
    assert not any(k.endswith("_H3") for k in tres)
    principal = analisis.analizar(carpeta, replicas=5)
    assert dos["2_lfs_ignicion_2008_2012_H1"]["efecto_pp"] == principal["H1"]["efecto_pp"]
    with pytest.raises(SourceError, match="not 4"):
        analisis.sensibilidad_lfs(carpeta, "4", replicas=5)
    salida = tmp_path / "s.json"
    assert main(["estudio", "sensibilidad", "--tablas", str(carpeta), "--numero", "3",
                 "--salida", str(salida), "--replicas", "5"]) == 0


def test_previous_class_without_burned_area_has_a_note(tmp_path):
    carpeta = _escribir(tmp_path / "t")
    a = analisis.analizar(carpeta, replicas=5)
    matriz = a["descriptivo"]["matriz_nivel1_persistente_k3"]
    fila = matriz["Cuerpo de agua"]  # no water pixel is burned in the synthetic table
    assert all(v.get("nota") == analisis.SIN_QUEMADO for v in fila.values())


def test_balance_uses_only_strata_with_enough_control(tmp_path):
    celda = int(h3n.latlng_to_cell(-35.0, -72.0, 7))
    filas, bal = [], []
    for anio, n_control, altura_c in ((2015, 500, 100.0), (2016, 50, 2000.0)):
        filas.append({"celda": celda, "anio": anio, "grupo": 1, "tramo": 0, "banda": 0, "c0": 1,
                      "c1": 1, "c2": 1, "area_m2": 100.0, "n_px": 10})
        filas.append({"celda": celda, "anio": anio, "grupo": 0, "tramo": 0, "banda": 0, "c0": 1,
                      "c1": 1, "c2": 1, "area_m2": 100.0, "n_px": n_control})
        for grupo, h in ((1, 100.0), (0, altura_c)):
            bal.append({"celda": celda, "anio": anio, "grupo": grupo, "tramo": 0,
                        "nivel1_previo": 1, "w": 100.0, "w_alt": 100.0 * h,
                        "w_alt2": 100.0 * h * h, "w_dist": 0.0, "w_dist2": 0.0})
    t = categorias.preparar(pd.DataFrame(filas))
    previa = t.l0.to_numpy() == 1
    e = estimador.Estimador(t, previa, np.zeros(len(t)), 1, 6, 4, np.unique(t.r4.to_numpy()))
    assert e.estratos_fuera == 1
    ruta = tmp_path / "balance.parquet"
    pq.write_table(pa.Table.from_pandas(pd.DataFrame(bal), preserve_index=False), ruta)
    r = analisis._balance(ruta, e)
    assert r["altitud_media_control_est_m"] == 100.0  # the 2016 stratum, with 50 px, is out
