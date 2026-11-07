import numpy as np
import pytest

from nadir import banco
from nadir.__main__ import main
from nadir.capas import cobertura
from nadir.errors import ProvenanceError, SourceError
from sinteticos import exportar

BANDAS = tuple(f"classification_{a}" for a in cobertura.ANIOS_ESTUDIO)


def preparar(bruto, imagen, bandas=BANDAS, nombre="drive-001.zip", **opciones):
    carpeta = bruto / cobertura.CARPETA
    banco.preparar(carpeta, cobertura.FUENTE)
    exportar(carpeta / nombre, cobertura.PREFIJO, imagen, bandas, lado=4, **opciones)
    banco.registrar(carpeta, nombre, "2012-2024", "earthengine:" + cobertura.ACTIVO, False)
    return carpeta


def imagen_valida(alto=6, ancho=9):
    imagen = np.full((len(BANDAS), alto, ancho), 66, dtype=np.uint8)
    imagen[:, 0, :] = 255
    imagen[:, 1, :3] = 3
    imagen[-1, 2, :] = 24
    imagen[:, 3, 0] = 0
    return imagen


def test_band_names_and_years():
    assert cobertura.banda(2012) == "classification_2012"
    with pytest.raises(SourceError, match="2012-2024"):
        cobertura.banda(2011)
    with pytest.raises(SourceError, match="2012-2024"):
        cobertura.banda(2025)


def test_source_header_is_the_one_in_the_bank():
    assert cobertura.FUENTE.conjunto == (
        "Cobertura y Uso del Suelo, mapas anuales, Colección 2.0 (1999-2024)"
    )
    assert cobertura.FUENTE.licencia == "MapBiomas-Chile"


def test_level_1_covers_every_legend_code_once():
    assert set(cobertura.NIVEL_1.values()) == {
        "formacion_boscosa",
        "formacion_natural_no_boscosa",
        "agropecuaria_y_silvicultura",
        "area_sin_vegetacion",
        "cuerpo_de_agua",
        "no_observado",
    }
    assert cobertura.SIN_DATO not in cobertura.NIVEL_1
    assert cobertura.FUERA not in cobertura.NIVEL_1


def test_describe_reports_area_by_code(tmp_path):
    imagen = imagen_valida()
    preparar(tmp_path, imagen)
    with cobertura.abrir(tmp_path) as mosaico:
        informe = cobertura.describir(mosaico, (2012, 2024))
        areas = mosaico.grilla.area_filas_m2(0, 6)
    assert informe.pixeles[2012] == {255: 9, 3: 3, 0: 1, 66: 41}
    assert informe.pixeles[2024] == {255: 9, 3: 3, 0: 1, 24: 9, 66: 32}
    esperado = (3 * areas[1]) / 10_000
    assert informe.hectareas[2012][3] == pytest.approx(esperado, rel=1e-12)
    total = sum(informe.hectareas[2024].values())
    assert total == pytest.approx(float(np.sum(areas)) * 9 / 10_000, rel=1e-12)


def test_unknown_code_is_refused(tmp_path):
    imagen = imagen_valida()
    imagen[5, 4, 4] = 42
    preparar(tmp_path, imagen)
    with cobertura.abrir(tmp_path) as mosaico, pytest.raises(SourceError, match=r"\[42\]"):
        cobertura.describir(mosaico)


def test_wrong_band_list_is_refused(tmp_path):
    preparar(tmp_path, imagen_valida()[:12], bandas=BANDAS[:12])
    with pytest.raises(SourceError, match="bands"):
        cobertura.abrir(tmp_path)


def test_wrong_no_data_is_refused(tmp_path):
    preparar(tmp_path, imagen_valida(), sin_dato=0)
    with pytest.raises(SourceError, match="no-data value 0"):
        cobertura.abrir(tmp_path)


def test_a_changed_zip_is_refused_before_reading(tmp_path):
    carpeta = preparar(tmp_path, imagen_valida())
    ruta = carpeta / "drive-001.zip"
    datos = ruta.read_bytes()
    ruta.write_bytes(datos[:-1] + bytes([datos[-1] ^ 0xFF]))
    with pytest.raises(ProvenanceError, match="sha256"):
        cobertura.abrir(tmp_path)


def test_cli_check_prints_areas(tmp_path, capsys):
    preparar(tmp_path, imagen_valida())
    assert main(["cobertura", "comprobar", "--bruto", str(tmp_path), "--anios", "2024"]) == 0
    salida = capsys.readouterr().out
    assert "12 tiles" not in salida
    assert "6 tiles" in salida
    assert "2024:" in salida
    assert "24 area_sin_vegetacion" in salida
    assert "  0 no_data" in salida
    assert "255" not in salida.split("2024:")[1]


def test_cli_refuses_years_outside_the_export(tmp_path, capsys):
    preparar(tmp_path, imagen_valida())
    assert main(["cobertura", "comprobar", "--bruto", str(tmp_path), "--anios", "2025"]) == 1
    assert "years outside 2012-2024: [2025]" in capsys.readouterr().err


def test_cli_builds_the_cell_map_once(tmp_path, capsys):
    preparar(tmp_path, imagen_valida())
    salida = tmp_path / "mapa"
    argumentos = ["celdas", "construir", "--bruto", str(tmp_path), "--resolucion", "8"]
    assert main([*argumentos, "--salida", str(salida)]) == 0
    assert "cells at resolution 8" in capsys.readouterr().out
    assert (salida / "mapa.json").exists()
    assert main([*argumentos, "--salida", str(salida)]) == 1
    assert "refusing to overwrite" in capsys.readouterr().err
