import shutil

import pytest

from nadir.__main__ import main
from nadir.errors import SourceError
from nadir.estudio import trayectorias
from test_estudio_analisis import _escribir


def _carpeta(tmp_path):
    carpeta = _escribir(tmp_path / "t")
    for nombre in ("k2", "k5"):
        shutil.copy(carpeta / "k3.parquet", carpeta / f"{nombre}.parquet")
    return carpeta


def test_each_row_adds_up_and_years_are_split(tmp_path):
    r = trayectorias.trayectorias(_carpeta(tmp_path))
    assert set(r) == {"nota", "trayectorias_k2", "trayectorias_k3", "trayectorias_k5",
                      "matriz_k3_por_anio"}
    for previa, fila in r["trayectorias_k3"].items():
        quemado = sum(v["quemado_pct"] for v in fila["destino"].values())
        control = sum(v["control_pct"] for v in fila["destino"].values())
        assert quemado == pytest.approx(100.0, abs=0.05), previa
        assert control == pytest.approx(100.0, abs=0.05), previa
        assert fila["area_quemada_en_estratos_ha"] > 0
    assert set(r["matriz_k3_por_anio"]) <= set(range(2013, 2022))
    assert r["trayectorias_k2"] == r["trayectorias_k3"]  # same synthetic table, same years


def test_k3_is_required(tmp_path):
    carpeta = _carpeta(tmp_path)
    (carpeta / "k3.parquet").unlink()
    with pytest.raises((SourceError, FileNotFoundError)):
        trayectorias.trayectorias(carpeta)


def test_command_line(tmp_path, capsys):
    carpeta = _carpeta(tmp_path)
    salida = tmp_path / "tr.json"
    assert main(["estudio", "trayectorias", "--tablas", str(carpeta), "--salida", str(salida)]) == 0
    assert "matrix by fire year" in capsys.readouterr().out
    assert salida.exists()
