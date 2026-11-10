import h3.api.numpy_int as h3n
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from nadir.__main__ import main
from nadir.errors import SourceError
from nadir.estudio import exploratorio


def _u5(carpeta, filas):
    carpeta.mkdir()
    columnas = {k: [f[k] for f in filas] for k in filas[0]}
    tipos = {"celda": np.int64, "anio": np.int16, "area_m2": np.float64, "n_px": np.int64}
    tabla = pa.table({k: np.asarray(v, dtype=tipos.get(k, np.uint8)) for k, v in columnas.items()})
    pq.write_table(tabla, carpeta / "u5.parquet")
    return carpeta


def _fila(celda, grupo, urbano, area, n, m24=0, banda=0, c0=3):
    codigo = 24 if urbano else c0
    return {"celda": celda, "anio": 2015, "grupo": grupo, "tramo": 0, "banda": banda, "m24": m24,
            "c0": c0, "c1": codigo, "c2": codigo, "area_m2": area, "n_px": n}


def test_effect_and_symmetric_mask(tmp_path):
    celda = int(h3n.latlng_to_cell(-35.0, -72.0, 7))
    filas = [
        _fila(celda, 1, True, 100.0, 10), _fila(celda, 1, False, 300.0, 30),
        _fila(celda, 0, True, 10.0, 60), _fila(celda, 0, False, 190.0, 60),
        _fila(celda, 1, True, 999.0, 99, m24=1), _fila(celda, 0, True, 1.0, 9, m24=1),
        _fila(celda, 1, True, 500.0, 50, c0=24),  # previously urban: outside the population
    ]
    r = exploratorio.urbanizacion(_u5(tmp_path / "t", filas), replicas=5)
    bosque = r["por_clase_previa"]["Formación boscosa"]
    assert bosque["p_quemado"] == 25.0
    assert bosque["p_control_estandarizado"] == 5.0
    assert bosque["efecto_pp"] == 20.0
    assert "menos de 30 bloques" in bosque["nota"]
    assert r["fuera_por_mascara_simetrica_ha_sumado_por_anio"] == 0.1
    assert r["por_clase_previa"]["Formación natural no boscosa"]["nota"] == (
        "sin área quemada comparable")
    assert r["interfaz_descriptivo"]["cerca_menos_de_2_km"]["diferencia_pp"] == 20.0
    assert "sin" in r["interfaz_descriptivo"]["lejos_2_km_o_mas"]["nota"]
    assert r["etiqueta"].startswith("exploratorio") and r["tabla"] == "u5"
    assert r["evaluable"] is True


def test_not_evaluable_when_no_burned_pixel_is_ever_code_24(tmp_path):
    celda = int(h3n.latlng_to_cell(-35.0, -72.0, 7))
    filas = [_fila(celda, 1, False, 100.0, 10), _fila(celda, 0, True, 10.0, 200),
             _fila(celda, 0, False, 190.0, 200)]
    r = exploratorio.urbanizacion(_u5(tmp_path / "t", filas), replicas=5)
    assert r["evaluable"] is False and "no evaluable" in r["motivo"]
    assert "por_clase_previa" not in r
    assert r["quemado_con_codigo_24_ha_sumado_por_anio"] == 0.0


def test_needs_the_exploratory_table(tmp_path):
    carpeta = tmp_path / "t"
    carpeta.mkdir()
    pq.write_table(pa.table({"celda": np.array([1], dtype=np.int64)}), carpeta / "u5.parquet")
    with pytest.raises(SourceError, match="not the exploratory"):
        exploratorio.urbanizacion(carpeta, replicas=5)


def test_command_line(tmp_path, capsys):
    rng = np.random.default_rng(3)
    celdas = [int(h3n.latlng_to_cell(a, b, 7))
              for a, b in zip(rng.uniform(-38, -33, 300), rng.uniform(-73, -71, 300), strict=True)]
    filas = [_fila(int(rng.choice(celdas)), int(rng.choice([0, 0, 1])), bool(rng.random() < 0.1),
                   float(rng.uniform(1e4, 1e5)), int(rng.integers(50, 300)),
                   banda=int(rng.integers(0, 2)), c0=int(rng.choice([3, 12, 9])))
             for _ in range(4000)]
    carpeta = _u5(tmp_path / "t", filas)
    salida = tmp_path / "u.json"
    args = ["estudio", "urbanizacion", "--tablas", str(carpeta), "--salida", str(salida),
            "--replicas", "20"]
    assert main(args) == 0
    assert "exploratory urbanisation" in capsys.readouterr().out
    import json

    r = json.loads(salida.read_text(encoding="utf-8"))
    assert set(r["por_clase_previa"]) == {"Formación boscosa", "Formación natural no boscosa",
                                          "Agropecuaria y silvicultura", "Las tres clases"}
    assert "ic95_pp" in r["por_clase_previa"]["Las tres clases"]


def test_lfs_table_has_no_mask(tmp_path):
    celda = int(h3n.latlng_to_cell(-35.0, -72.0, 7))
    filas = [_fila(celda, 1, True, 100.0, 10), _fila(celda, 1, False, 300.0, 30),
             _fila(celda, 0, True, 10.0, 60), _fila(celda, 0, False, 190.0, 60)]
    for f in filas:
        del f["m24"]
    carpeta = _u5(tmp_path / "t", filas)
    (carpeta / "u5.parquet").rename(carpeta / "l5.parquet")
    r = exploratorio.urbanizacion(carpeta, replicas=5, tabla="l5")
    assert r["tabla"] == "l5" and r["evaluable"] is True
    assert r["por_clase_previa"]["Formación boscosa"]["efecto_pp"] == 20.0
    assert r["fuera_por_mascara_simetrica_ha_sumado_por_anio"] == 0.0
    with pytest.raises(SourceError, match="not k3"):
        exploratorio.urbanizacion(carpeta, replicas=5, tabla="k3")
