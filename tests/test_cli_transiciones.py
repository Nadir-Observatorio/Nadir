import zipfile

import numpy as np
import pyarrow.parquet as pq
import pytest

from nadir import banco
from nadir.__main__ import main
from nadir.capas import cobertura, fuego, lfs
from sinteticos import PASO, exportar
from test_lfs import FUENTE, agregar_fuego

OESTE, NORTE = -71.5, -33.5
ALTO, ANCHO = 12, 14


def _banco(raiz):
    rng = np.random.default_rng(2)
    cob = raiz / cobertura.CARPETA
    banco.preparar(cob, cobertura.FUENTE)
    imagen = rng.choice([3, 9, 12, 24, 66], size=(13, ALTO, ANCHO)).astype(np.uint8)
    bandas = tuple(cobertura.banda(a) for a in cobertura.ANIOS_ESTUDIO)
    exportar(cob / "c.zip", cobertura.PREFIJO, imagen, bandas, lado=8, oeste=OESTE, norte=NORTE)
    banco.registrar(cob, "c.zip", "2012-2024", "earthengine:x", False)
    fue = raiz / fuego.CARPETA
    banco.preparar(fue, fuego.FUENTE)
    anual = (rng.random((13, ALTO, ANCHO)) < 0.2).astype(np.uint8)
    mensual = anual * 2
    exportar(fue / "f.zip", fuego.PREFIJO_ANUAL, anual,
             tuple(fuego.banda_anual(a) for a in fuego.ANIOS), lado=8, oeste=OESTE, norte=NORTE)
    exportar(fue / "f.zip", fuego.PREFIJO_MENSUAL, mensual,
             tuple(fuego.banda_mensual(a) for a in fuego.ANIOS), lado=8, oeste=OESTE, norte=NORTE)
    banco.registrar(fue, "f.zip", "2013-2025", "earthengine:x", False)
    car = raiz / lfs.CARPETA
    car.mkdir(parents=True)
    banco.preparar(car, FUENTE)
    with zipfile.ZipFile(car / "FireScar_CL-VS_Valparaiso_2009-2018.zip", "w") as zf:
        agregar_fuego(zf, "r", "CL-VS", 9, 400, "20100110", np.ones((4, 4)),
                      OESTE + 3 * PASO, NORTE - 3 * PASO, 5e6)
    banco.registrar(car, "FireScar_CL-VS_Valparaiso_2009-2018.zip", "p", lfs.PAGINA, False)


def test_cli_builds_scars_and_tables(tmp_path, capsys):
    raiz = tmp_path / "bruto"
    _banco(raiz)
    mapa, cic = tmp_path / "celdas", tmp_path / "lfs"
    assert main(["celdas", "construir", "--bruto", str(raiz), "--resolucion", "8",
                 "--salida", str(mapa)]) == 0
    assert main(["lfs", "construir", "--bruto", str(raiz), "--salida", str(cic)]) == 0
    assert "1 scars read; 1 touch the grid; 0 kept for the comparison" in capsys.readouterr().out
    base = ["transiciones", "tabular", "--bruto", str(raiz), "--celdas", str(mapa),
            "--desfases=-1,2,3", "--anios", "2013", "2021"]
    sin = tmp_path / "sin.parquet"
    con = tmp_path / "con.parquet"
    assert main([*base, "--salida", str(sin)]) == 0
    assert main([*base, "--cicatrices", str(cic), "--excluir", "ignicion_2008_2012",
                 "--salida", str(con)]) == 0
    texto = capsys.readouterr().out
    assert "fire years 2013-2021, offsets (-1, 2, 3)" in texto
    meta = pq.read_table(con).schema.metadata
    assert meta[b"excluir_bits"] == str(lfs.BIT_IGNICION_2008_2012).encode()
    assert b"cicatrices_sha256" in meta[b"procedencia"]
    area_sin = sum(pq.read_table(sin).column("area_m2").to_pylist())
    area_con = sum(pq.read_table(con).column("area_m2").to_pylist())
    assert area_con < area_sin
    fuera = float(meta[b"excluido_por_cicatrices_m2"])
    assert area_con + fuera == pytest.approx(area_sin, rel=1e-9)
    assert main([*base, "--salida", str(sin)]) == 1
    assert main([*base[:-3], "--anios", "2021", "2013", "--salida", str(tmp_path / "x")]) == 1
    assert "first and last" in capsys.readouterr().err
    assert main([*base[:6], "--desfases=-1,a", "--anios", "2013", "2014",
                 "--salida", str(tmp_path / "y")]) == 1
