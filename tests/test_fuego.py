import numpy as np
import pytest

from nadir import banco
from nadir.__main__ import main
from nadir.capas import cobertura, fuego
from nadir.errors import SourceError
from sinteticos import exportar

ANUAL = tuple(fuego.banda_anual(a) for a in fuego.ANIOS)
MENSUAL = tuple(fuego.banda_mensual(a) for a in fuego.ANIOS)


def imagenes(alto=5, ancho=6):
    mensual = np.zeros((len(fuego.ANIOS), alto, ancho), dtype=np.uint8)
    mensual[:, :, 0] = 255
    mensual[0, 1, 1] = 2
    mensual[4, 1, 1] = 12
    mensual[11, 2:4, 2:5] = 2
    mensual[11, 4, 5] = 10
    anual = np.where(mensual == 255, 255, (mensual > 0).astype(np.uint8)).astype(np.uint8)
    return anual, mensual


def preparar(bruto, anual, mensual, **opciones):
    carpeta = bruto / fuego.CARPETA
    banco.preparar(carpeta, fuego.FUENTE)
    ruta = carpeta / "drive.zip"
    exportar(ruta, fuego.PREFIJO_ANUAL, anual, ANUAL, lado=4, **opciones)
    exportar(ruta, fuego.PREFIJO_MENSUAL, mensual, MENSUAL, lado=4)
    banco.registrar(carpeta, "drive.zip", "2013-2025", "earthengine:FIRE", False)
    return carpeta


def test_band_names():
    assert fuego.banda_anual(2024) == "burned_area_2024"
    assert fuego.banda_mensual(2013) == "burned_monthly_2013"
    with pytest.raises(SourceError, match="2013-2025"):
        fuego.banda_anual(2012)


def test_source_header_is_the_one_in_the_bank():
    assert fuego.FUENTE.conjunto == (
        "Area Quemada (MapBiomas Fuego Chile), Coleccion 1 (2013-2025), anual y mensual"
    )


def test_describe_reports_burned_area_by_year_and_month(tmp_path):
    preparar(tmp_path, *imagenes())
    anual, mensual = fuego.abrir(tmp_path)
    with anual, mensual:
        informe = fuego.describir(anual, mensual, (2013, 2024))
        areas = anual.grilla.area_filas_m2(0, 5)
    assert informe.pixeles_anio[2024] == {255: 5, 0: 18, 1: 7}
    esperado = (3 * areas[2] + 3 * areas[3] + areas[4]) / 10_000
    assert informe.hectareas_anio[2024] == pytest.approx(esperado, rel=1e-12)
    assert set(informe.hectareas_mes[2024]) == {2, 10}
    assert sum(informe.hectareas_mes[2024].values()) == pytest.approx(esperado, rel=1e-12)
    assert informe.hectareas_mes[2013] == {2: pytest.approx(areas[1] / 10_000, rel=1e-12)}


def test_frequency_counts_burned_years(tmp_path):
    preparar(tmp_path, *imagenes())
    anual, mensual = fuego.abrir(tmp_path)
    with anual, mensual:
        veces = fuego.frecuencia(anual, 0, 5)
    assert veces[1, 1] == 2
    assert veces[2, 2] == 1
    assert veces[0, 3] == 0
    assert np.all(veces[:, 0] == 255)


@pytest.mark.parametrize(
    ("cambio", "mensaje"),
    [
        (lambda a, m: a.__setitem__((11, 0, 3), 1), "disagree on burned"),
        (lambda a, m: m.__setitem__((11, 0, 3), 13), "monthly values"),
        (lambda a, m: a.__setitem__((11, 0, 3), 7), "annual values"),
        (lambda a, m: m.__setitem__((11, 0, 0), 0), "disagree on the seven regions"),
    ],
)
def test_inconsistent_exports_are_refused(tmp_path, cambio, mensaje):
    anual, mensual = imagenes()
    cambio(anual, mensual)
    preparar(tmp_path, anual, mensual)
    a, m = fuego.abrir(tmp_path)
    with a, m, pytest.raises(SourceError, match=mensaje):
        fuego.describir(a, m, (2024,))


def test_grids_must_match(tmp_path):
    preparar(tmp_path, *imagenes(), norte=-34.0)
    with pytest.raises(SourceError, match="different grids"):
        fuego.abrir(tmp_path)


def test_cli_reports_and_compares_with_land_cover(tmp_path, capsys):
    preparar(tmp_path, *imagenes())
    cob = tmp_path / cobertura.CARPETA
    banco.preparar(cob, cobertura.FUENTE)
    imagen = np.full((13, 4, 4), 66, dtype=np.uint8)
    exportar(
        cob / "c.zip",
        cobertura.PREFIJO,
        imagen,
        tuple(cobertura.banda(a) for a in cobertura.ANIOS_ESTUDIO),
        lado=4,
        oeste=-72.0 - 3 * 0.0002694945852358564,
    )
    banco.registrar(cob, "c.zip", "2012-2024", "earthengine:LULC", False)
    assert main(["fuego", "comprobar", "--bruto", str(tmp_path), "--anios", "2024"]) == 0
    salida = capsys.readouterr().out
    assert "2024:" in salida
    assert "2: " in salida and "10: " in salida
    assert "aligned, offset of 3 columns and 0 rows" in salida


def test_cli_without_land_cover_says_nothing_about_it(tmp_path, capsys):
    preparar(tmp_path, *imagenes())
    assert main(["fuego", "comprobar", "--bruto", str(tmp_path), "--anios", "2013"]) == 0
    assert "Land cover" not in capsys.readouterr().out


def imagen_cicatrices():
    """Annual 2024 with three scars: a diagonal chain, a block cut by the strip edge, a dot."""
    mensual = np.zeros((len(fuego.ANIOS), 9, 8), dtype=np.uint8)
    mensual[:, :, 7] = 255
    b = fuego.ANIOS.index(2024)
    for i in range(4):
        mensual[b, i, i] = 2  # diagonal: one scar through corners
    mensual[b, 3:6, 5:7] = 3  # block across rows 3 to 5
    mensual[b, 8, 0] = 11  # single pixel
    anual = np.where(mensual == 255, 255, (mensual > 0).astype(np.uint8)).astype(np.uint8)
    return anual, mensual


@pytest.mark.parametrize("alto", [1, 2, 4, 1024])
def test_scars_do_not_depend_on_the_strip_height(tmp_path, alto):
    preparar(tmp_path, *imagen_cicatrices())
    anual, mensual = fuego.abrir(tmp_path)
    with anual, mensual:
        lista = fuego.cicatrices(anual, 2024, alto)
        informe = fuego.describir(anual, mensual, (2024,))
        g = anual.grilla
    assert [(c.numero, c.pixeles) for c in lista] == [(1, 4), (2, 6), (3, 1)]
    diagonal, bloque, punto = lista
    assert sum(c.hectareas for c in lista) == pytest.approx(informe.hectareas_anio[2024])
    assert (diagonal.norte, diagonal.oeste) == (g.norte, g.oeste)
    assert diagonal.sur == pytest.approx(g.norte - 4 * g.paso)
    assert diagonal.este == pytest.approx(g.oeste + 4 * g.paso)
    assert bloque.oeste == pytest.approx(g.oeste + 5 * g.paso)
    assert punto.sur == pytest.approx(g.norte - 9 * g.paso)
    assert 0.11 < diagonal.diagonal_km < 0.17  # 4 x 4 pixels of about 30 x 25 m
    assert all(c.anio == 2024 for c in lista)


def test_scars_join_through_corners_across_strips(tmp_path):
    mensual = np.zeros((len(fuego.ANIOS), 4, 4), dtype=np.uint8)
    b = fuego.ANIOS.index(2013)
    mensual[b, 1, 2] = 1
    mensual[b, 2, 1] = 1  # touches the first only by a corner, on the next strip
    mensual[b, 2, 3] = 1  # and this one by the other corner
    anual = (mensual > 0).astype(np.uint8)
    preparar(tmp_path, anual, mensual)
    anual_m, mensual_m = fuego.abrir(tmp_path)
    with anual_m, mensual_m:
        assert [c.pixeles for c in fuego.cicatrices(anual_m, 2013, 2)] == [3]
        assert fuego.cicatrices(anual_m, 2014, 2) == []
        with pytest.raises(Exception, match="strip height"):
            fuego.cicatrices(anual_m, 2013, 0)


def test_scars_csv_is_written_once(tmp_path, capsys):
    preparar(tmp_path, *imagen_cicatrices())
    salida = tmp_path / "out" / "cicatrices.csv"
    argumentos = ["fuego", "cicatrices", "--bruto", str(tmp_path), "--salida", str(salida)]
    assert main([*argumentos, "--anios", "2024", "2013"]) == 0
    texto = capsys.readouterr().out
    assert "2013: no scars" in texto
    assert "2024: 3 scars" in texto
    filas = salida.read_text(encoding="utf-8").splitlines()
    assert filas[0] == "anio,numero,pixeles,hectareas,norte,sur,oeste,este,diagonal_km"
    assert len(filas) == 4
    assert filas[1].startswith("2024,1,4,")
    assert b"\r" not in salida.read_bytes()
    assert main([*argumentos, "--anios", "2024"]) == 1
    assert "refusing to overwrite" in capsys.readouterr().err
    with pytest.raises(Exception, match="refusing to overwrite"):
        fuego.escribir_cicatrices(salida, [])
