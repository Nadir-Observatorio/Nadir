import json
import math

import numpy as np
import pytest
import rasterio
from rasterio.warp import transform

from nadir import banco, exportacion
from nadir.__main__ import main
from nadir.capas import cobertura, interfaz_wui
from nadir.errors import SourceError
from sinteticos import exportar, tif

BANDAS = tuple(f"classification_{a}" for a in cobertura.ANIOS_ESTUDIO)
PASO = 0.02
OESTE, NORTE = -72.0, -35.5
ALTO, ANCHO = 40, 30
LADO_M = 500.0
UTM = "EPSG:32719"


def _cobertura(bruto):
    carpeta = bruto / cobertura.CARPETA
    banco.preparar(carpeta, cobertura.FUENTE)
    imagen = np.full((len(BANDAS), ALTO, ANCHO), 3, dtype=np.uint8)
    imagen[:, 0, :6] = cobertura.FUERA
    imagen[:, -1, -3:] = cobertura.FUERA
    exportar(carpeta / "drive-001.zip", cobertura.PREFIJO, imagen, BANDAS, lado=16, paso=PASO,
             oeste=OESTE, norte=NORTE)
    banco.registrar(carpeta, "drive-001.zip", "2012-2024", "earthengine:x", False)
    return imagen[cobertura.ANIOS_ESTUDIO.index(interfaz_wui.ANIO_MASCARA)]


def _origen_utm():
    """Corner of the WUI rasters: they cover all but the eastern fifth of the grid."""
    xs, ys = transform("EPSG:4326", UTM, [OESTE - 0.05], [NORTE + 0.05])
    return math.floor(xs[0] / LADO_M) * LADO_M, math.ceil(ys[0] / LADO_M) * LADO_M


def _rasters(bruto, semilla=4):
    rng = np.random.default_rng(semilla)
    oeste_m, norte_m = _origen_utm()
    ancho = round(0.85 * ANCHO * PASO * 90_000 / LADO_M)
    alto = round((ALTO * PASO + 0.1) * 111_000 / LADO_M)
    valores = []
    for _ in interfaz_wui.ARCHIVOS:
        v = rng.integers(1, 13, (alto, ancho)) / 12.0
        v[rng.random((alto, ancho)) < 0.5] = np.nan
        valores.append(v.astype(np.float32))
    carpeta = bruto / interfaz_wui.CARPETA
    banco.preparar(carpeta, banco.Fuente("Zenodo", "WUI Chile", "https://doi.org/x",
                                         "CC-BY-4.0", "x"))
    for nombre, v in zip(interfaz_wui.ARCHIVOS, valores, strict=True):
        (carpeta / nombre).write_bytes(
            tif(v[None], ("intensity",), oeste=oeste_m, norte=norte_m, paso=LADO_M, crs=UTM,
                sin_dato=np.nan, tipo="float32"))
        banco.registrar(carpeta, nombre, "2025", "https://zenodo.org/x", True)
    return valores, oeste_m, norte_m


def _bruta(valores, oeste_m, norte_m, g):
    """Band of every export pixel from the source pixel that holds its centre."""
    lon, lat = np.meshgrid(g.longitudes(0, g.ancho), g.latitudes(0, g.alto))
    xs, ys = transform("EPSG:4326", UTM, lon.ravel().tolist(), lat.ravel().tolist())
    col = np.floor((np.array(xs) - oeste_m) / LADO_M).astype(int)
    fil = np.floor((norte_m - np.array(ys)) / LADO_M).astype(int)
    alto, ancho = valores[0].shape
    dentro = (col >= 0) & (col < ancho) & (fil >= 0) & (fil < alto)
    banda = np.ones(col.size, dtype=np.uint8)
    for v in valores:
        x = np.full(col.size, np.nan)
        x[dentro] = v[fil[dentro], col[dentro]]
        banda[x > 0.5] = 0
    return banda.reshape(g.alto, g.ancho), dentro.reshape(g.alto, g.ancho)


def test_classification_needs_a_majority_and_ignores_no_data():
    a = np.array([np.nan, 0.5, 7 / 12, 0.25, np.nan], dtype=np.float32)
    b = np.array([np.nan, 0.5, np.nan, 1.0, 0.0], dtype=np.float32)
    assert interfaz_wui.clasificar(a, b).tolist() == [1, 1, 0, 0, 1]


def test_build_takes_the_source_pixel_under_each_centre(tmp_path, monkeypatch):
    monkeypatch.setattr(interfaz_wui, "FILAS_FRANJA", 7)  # several strips
    mascara = _cobertura(tmp_path / "bruto")
    valores, oeste_m, norte_m = _rasters(tmp_path / "bruto")
    with cobertura.abrir(tmp_path / "bruto") as mosaico:
        registro = interfaz_wui.construir(tmp_path / "bruto", mosaico, tmp_path / "wui")
        g = mosaico.grilla
    esperado, cubierto = _bruta(valores, oeste_m, norte_m, g)
    assert cubierto.any() and not cubierto.all()  # part of the grid lies outside the rasters
    fuera = mascara == cobertura.FUERA
    esperado[fuera] = interfaz_wui.FUERA
    obtenido = interfaz_wui.leer_banda(tmp_path / "wui", g, 0, g.alto)
    assert (obtenido == esperado).all()
    assert (obtenido == 0).any() and (obtenido == 1).any()
    assert registro["pixeles_dentro"] == int((~fuera).sum())
    assert registro["pixeles_interfaz"] == int((esperado == 0).sum())
    areas = g.area_filas_m2(0, g.alto)[:, None]
    assert registro["area_interfaz_ha"] == round(float(((esperado == 0) * areas).sum()) / 1e4, 1)
    guardado = json.loads((tmp_path / "wui" / "registro_wui.json").read_text(encoding="utf-8"))
    assert guardado == registro
    assert set(guardado["fuentes"]) == set(interfaz_wui.ARCHIVOS)
    assert guardado["crs_fuente"] == UTM
    assert not list((tmp_path / "wui").glob("*.part"))
    assert (interfaz_wui.leer_banda(tmp_path / "wui", g, 3, 4) == obtenido[3:7]).all()


def test_build_checks_the_bank_and_never_overwrites(tmp_path):
    _cobertura(tmp_path / "bruto")
    banco.preparar(tmp_path / "bruto" / interfaz_wui.CARPETA,
                   banco.Fuente("Zenodo", "WUI", "https://doi.org/x", "CC-BY-4.0", "x"))
    with cobertura.abrir(tmp_path / "bruto") as mosaico, pytest.raises(
        SourceError, match="not recorded"
    ):
        interfaz_wui.construir(tmp_path / "bruto", mosaico, tmp_path / "wui")


def test_never_overwrites_and_checks_the_grid(tmp_path):
    _cobertura(tmp_path / "bruto")
    _rasters(tmp_path / "bruto")
    with cobertura.abrir(tmp_path / "bruto") as mosaico:
        interfaz_wui.construir(tmp_path / "bruto", mosaico, tmp_path / "wui")
        with pytest.raises(SourceError, match="overwrite"):
            interfaz_wui.construir(tmp_path / "bruto", mosaico, tmp_path / "wui")
        g = mosaico.grilla
    otra = exportacion.Grilla(g.oeste + 1, g.norte, g.paso, g.ancho, g.alto)
    with pytest.raises(SourceError, match="another grid"):
        interfaz_wui.leer_banda(tmp_path / "wui", otra, 0, 1)


def test_rasters_on_different_grids_are_refused(tmp_path):
    _cobertura(tmp_path / "bruto")
    _rasters(tmp_path / "bruto")
    ruta = tmp_path / "bruto" / interfaz_wui.CARPETA / interfaz_wui.ARCHIVOS[1]
    with rasterio.open(ruta) as r:
        v = r.read()
        oeste_m, norte_m = r.transform.c, r.transform.f
    otra = tmp_path / "otra"
    otra.mkdir()
    carpeta = otra / interfaz_wui.CARPETA
    banco.preparar(carpeta, banco.Fuente("Zenodo", "WUI", "https://doi.org/x", "CC-BY-4.0", "x"))
    for nombre, dx in zip(interfaz_wui.ARCHIVOS, (0.0, LADO_M), strict=True):
        (carpeta / nombre).write_bytes(tif(v, ("intensity",), oeste=oeste_m + dx, norte=norte_m,
                                           paso=LADO_M, crs=UTM, sin_dato=np.nan,
                                           tipo="float32"))
        banco.registrar(carpeta, nombre, "2025", "https://zenodo.org/x", True)
    with cobertura.abrir(tmp_path / "bruto") as mosaico, pytest.raises(
        SourceError, match="same grid"
    ):
        interfaz_wui.construir(otra, mosaico, tmp_path / "wui")


def test_command_line_builds_the_band(tmp_path, capsys):
    _cobertura(tmp_path / "bruto")
    _rasters(tmp_path / "bruto")
    codigo = main(["interfaz", "wui", "--bruto", str(tmp_path / "bruto"),
                   "--salida", str(tmp_path / "wui")])
    assert codigo == 0
    assert "pixels are interface" in capsys.readouterr().out
    assert (tmp_path / "wui" / "banda_wui.u8").stat().st_size == ALTO * ANCHO
