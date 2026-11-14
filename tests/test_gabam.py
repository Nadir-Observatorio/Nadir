import json

import numpy as np
import pytest

from nadir import banco
from nadir.capas import cobertura, gabam
from nadir.errors import SourceError
from sinteticos import PASO, exportar

BANDAS = tuple(cobertura.banda(a) for a in cobertura.ANIOS_ESTUDIO)
OESTE, NORTE = -71.5, -33.5
ALTO, ANCHO = 20, 24
PASO_G = PASO * 0.93  # a finer grid, offset from the land cover grid
OESTE_G, NORTE_G = OESTE - 0.37 * PASO, NORTE + 0.21 * PASO
ALTO_G, ANCHO_G = 19, 30  # leaves the southern rows of the land cover grid uncovered


def _banco(raiz, semilla=3, valores=(0, 1)):
    rng = np.random.default_rng(semilla)
    carpeta = raiz / cobertura.CARPETA
    banco.preparar(carpeta, cobertura.FUENTE)
    cob = np.full((len(BANDAS), ALTO, ANCHO), 3, dtype=np.uint8)
    cob[:, 0, :5] = cobertura.FUERA
    exportar(carpeta / "c.zip", cobertura.PREFIJO, cob, BANDAS, lado=8, oeste=OESTE, norte=NORTE)
    banco.registrar(carpeta, "c.zip", "2012-2024", "earthengine:x", False)
    gdir = raiz / gabam.CARPETA
    banco.preparar(gdir, banco.Fuente("AIRCAS", "GABAM", "https://x", "CC0-1.0", "x"))
    imagenes = {}
    for anio in gabam.ANIOS:
        img = rng.choice(valores, size=(1, ALTO_G, ANCHO_G)).astype(np.uint8)
        img[0, :, -2:] = gabam.FUERA
        exportar(gdir / "g.zip", gabam.prefijo(anio), img, (gabam.banda(anio),), lado=7,
                 paso=PASO_G, oeste=OESTE_G, norte=NORTE_G, sin_dato=gabam.FUERA)
        imagenes[anio] = img[0]
    banco.registrar(gdir, "g.zip", "2013-2021", "earthengine:x", False)
    return cob[BANDAS.index(cobertura.banda(2020))], imagenes


def _esperado(img, mascara, g):
    lat = g.latitudes(0, g.alto)
    lon = g.longitudes(0, g.ancho)
    f = np.floor((NORTE_G - lat) / PASO_G).astype(int)
    c = np.floor((lon - OESTE_G) / PASO_G).astype(int)
    out = np.full((g.alto, g.ancho), gabam.FUERA, dtype=np.uint8)
    for i in range(g.alto):
        for j in range(g.ancho):
            if 0 <= f[i] < ALTO_G and 0 <= c[j] < ANCHO_G:
                out[i, j] = img[f[i], c[j]]
    out[mascara == cobertura.FUERA] = gabam.FUERA
    return out


def test_each_pixel_takes_the_gabam_pixel_under_its_centre(tmp_path, monkeypatch):
    monkeypatch.setattr(gabam, "FILAS_FRANJA", 6)
    mascara, imagenes = _banco(tmp_path / "bruto")
    with cobertura.abrir(tmp_path / "bruto") as m:
        registro = gabam.construir(tmp_path / "bruto", m, tmp_path / "g", cobertura.banda(2020))
        g = m.grilla
    for anio, img in imagenes.items():
        esperado = _esperado(img, mascara, g)
        obtenido = gabam.leer(tmp_path / "g", g, anio, 0, g.alto)
        assert (obtenido == esperado).all(), anio
        datos = registro["anios"][str(anio)]
        assert datos["pixeles_quemados"] == int((esperado == gabam.QUEMADO).sum())
        assert datos["pixeles_dentro"] == int((esperado != gabam.FUERA).sum())
    assert (gabam.leer(tmp_path / "g", g, 2013, 0, g.alto)[-1] == gabam.FUERA).all()
    guardado = json.loads((tmp_path / "g" / "registro_gabam.json").read_text(encoding="utf-8"))
    assert guardado == registro and "g.zip" in guardado["fuentes"]
    with cobertura.abrir(tmp_path / "bruto") as m, pytest.raises(SourceError, match="overwrite"):
        gabam.construir(tmp_path / "bruto", m, tmp_path / "g", cobertura.banda(2020))


def test_unexpected_values_stop_the_build(tmp_path):
    _banco(tmp_path / "bruto", valores=(0, 1, 7))
    with cobertura.abrir(tmp_path / "bruto") as m, pytest.raises(SourceError, match="unexpected"):
        gabam.construir(tmp_path / "bruto", m, tmp_path / "g", cobertura.banda(2020))


def test_years_outside_the_range_are_refused(tmp_path):
    _banco(tmp_path / "bruto")
    with pytest.raises(SourceError, match="2013-2021"):
        gabam.abrir(tmp_path / "bruto", 2022)
