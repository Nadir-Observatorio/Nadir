import hashlib
import json
import math

import numpy as np
import pytest

from nadir import exportacion
from nadir.__main__ import main
from nadir.capas import cobertura, interfaz
from nadir.errors import SourceError
from sinteticos import exportar

BANDAS = tuple(f"classification_{a}" for a in cobertura.ANIOS_ESTUDIO)
PASO_GRUESO = 0.02  # strips of 5 rows and a margin of 7 rows, so the strip logic is exercised


def preparar(bruto, imagen, paso=PASO_GRUESO):
    from nadir import banco

    carpeta = bruto / cobertura.CARPETA
    banco.preparar(carpeta, cobertura.FUENTE)
    exportar(
        carpeta / "drive-001.zip", cobertura.PREFIJO, imagen, BANDAS, lado=16, paso=paso,
        oeste=-72.0, norte=-35.5,
    )
    banco.registrar(carpeta, "drive-001.zip", "2012-2024", "earthengine:" + cobertura.ACTIVO, False)
    return carpeta


def imagen_con_interfaz(alto=40, ancho=30, semilla=3):
    rng = np.random.default_rng(semilla)
    imagen = np.full((len(BANDAS), alto, ancho), 3, dtype=np.uint8)
    for i in range(len(BANDAS)):
        filas = rng.integers(0, alto, 4)
        columnas = rng.integers(0, ancho, 4)
        imagen[i, filas, columnas] = interfaz.INFRAESTRUCTURA
    imagen[:, 0, :6] = cobertura.FUERA
    imagen[:, -1, -3:] = cobertura.FUERA
    return imagen


def fuerza_bruta(mapa, fila, columna, dy, dx):
    rr, cc = np.nonzero(mapa == interfaz.INFRAESTRUCTURA)
    if rr.size == 0:
        return math.inf
    return float(np.sqrt(((rr - fila) * dy) ** 2 + ((cc - columna) * dx) ** 2).min())


def test_spacing_follows_the_wgs84_ellipsoid():
    meridiano, normal = interfaz.radios(0.0)
    assert normal == pytest.approx(6378137.0)
    assert meridiano == pytest.approx(6335439.327, abs=1e-3)
    dy, dx = interfaz.espaciado(-36.0, 1 / 3600)
    assert dy == pytest.approx(30.83, abs=0.01)
    assert dx == pytest.approx(25.04, abs=0.01)
    assert interfaz.margen_filas(PASO_GRUESO) == math.ceil(10000 / (2 * dy * 36)) + 2


def test_distance_matches_brute_force_on_an_anisotropic_grid():
    rng = np.random.default_rng(7)
    ventana = rng.integers(0, 30, (25, 31)).astype(np.uint8)
    ventana[ventana > 2] = 3
    ventana[ventana <= 2] = interfaz.INFRAESTRUCTURA
    dy, dx = 31.0, 24.5
    metros = interfaz.distancia(ventana, dy, dx)
    for f in range(0, 25, 4):
        for c in range(0, 31, 5):
            assert metros[f, c] == pytest.approx(fuerza_bruta(ventana, f, c, dy, dx))


def test_window_without_interface_is_far_everywhere():
    metros = interfaz.distancia(np.full((3, 4), 3, dtype=np.uint8), 30.0, 25.0)
    assert np.isinf(metros).all()
    banda, dist40 = interfaz.codificar(metros, np.zeros((3, 4), dtype=bool))
    assert (banda == 1).all()
    assert (dist40 == interfaz.TOPE_PASOS).all()


def test_coding_of_band_steps_cap_and_outside():
    metros = np.array([[0.0, 19.9, 20.1, 1999.9, 2000.0, 9999.0, 10000.0, 50000.0]])
    fuera = np.zeros_like(metros, dtype=bool)
    fuera[0, -1] = True
    banda, dist40 = interfaz.codificar(metros, fuera)
    assert banda.tolist() == [[0, 0, 0, 0, 1, 1, 1, 255]]
    assert dist40.tolist() == [[0, 0, 1, 50, 50, 250, 250, 255]]


def test_build_is_exact_up_to_the_cap_across_strips(tmp_path):
    imagen = imagen_con_interfaz()
    preparar(tmp_path / "bruto", imagen)
    salida = tmp_path / "interfaz"
    with cobertura.abrir(tmp_path / "bruto") as mosaico:
        registro = interfaz.construir(mosaico, salida, (2012, 2016))
        g = mosaico.grilla
    assert registro["filas_por_franja"] == 5
    assert registro["margen_filas"] == 7
    for anio in (2012, 2016):
        mapa = imagen[cobertura.ANIOS_ESTUDIO.index(anio)]
        banda = interfaz.leer_banda(salida, g, anio, 0, g.alto)
        dist40 = interfaz.leer_dist40(salida, g, anio, 0, g.alto)
        for f in range(g.alto):
            franja = f // 5
            n = min(5, g.alto - 5 * franja)
            lat_c = float(g.latitudes(5 * franja, n).mean())
            dy, dx = interfaz.espaciado(lat_c, g.paso)
            for c in range(g.ancho):
                if mapa[f, c] == cobertura.FUERA:
                    assert banda[f, c] == 255
                    assert dist40[f, c] == 255
                    continue
                d = fuerza_bruta(mapa, f, c, dy, dx)
                if d <= interfaz.TOPE_M:
                    assert dist40[f, c] == min(round(d / 40.0), 250)
                    assert banda[f, c] == (0 if d < 2000.0 else 1)
                else:
                    assert dist40[f, c] == 250
                    assert banda[f, c] == 1
        datos = registro["anios"][str(anio)]
        assert datos["sha256_banda"] == hashlib.sha256(banda.tobytes()).hexdigest()
        assert datos["sha256_dist40"] == hashlib.sha256(dist40.tobytes()).hexdigest()
        assert datos["pixeles_dentro"] == int((banda != 255).sum())
        assert datos["pixeles_cerca"] == int((banda == 0).sum())
    mapa = imagen[cobertura.ANIOS_ESTUDIO.index(2012)]
    fuera = mapa == cobertura.FUERA
    franja = 0.0
    areas = g.area_filas_m2(0, g.alto)
    rr, cc = np.nonzero(fuera)
    for f in range(g.alto):
        n = min(5, g.alto - 5 * (f // 5))
        dy, dx = interfaz.espaciado(float(g.latitudes(5 * (f // 5), n).mean()), g.paso)
        for c in range(g.ancho):
            cerca = np.sqrt(((rr - f) * dy) ** 2 + ((cc - c) * dx) ** 2).min() < 2000
            if not fuera[f, c] and cerca:
                franja += areas[f]
    assert franja > 0
    assert registro["franja_borde_2km_ha"] == round(franja / 1e4, 1)
    guardado = json.loads((salida / "registro.json").read_text(encoding="utf-8"))
    assert guardado["anios"] == registro["anios"]
    assert not list(salida.glob("*.part"))


def test_build_never_overwrites_and_checks_years(tmp_path):
    preparar(tmp_path / "bruto", imagen_con_interfaz())
    salida = tmp_path / "interfaz"
    with cobertura.abrir(tmp_path / "bruto") as mosaico:
        interfaz.construir(mosaico, salida, (2013,))
        with pytest.raises(SourceError, match="overwrite"):
            interfaz.construir(mosaico, salida, (2013,))
        with pytest.raises(SourceError, match="2012-2024"):
            interfaz.construir(mosaico, tmp_path / "otra", (2011,))


def test_reading_on_another_grid_fails(tmp_path):
    preparar(tmp_path / "bruto", imagen_con_interfaz())
    salida = tmp_path / "interfaz"
    with cobertura.abrir(tmp_path / "bruto") as mosaico:
        interfaz.construir(mosaico, salida, (2014,))
        g = mosaico.grilla
    otra = exportacion.Grilla(g.oeste + g.paso, g.norte, g.paso, g.ancho, g.alto)
    with pytest.raises(SourceError, match="another grid"):
        interfaz.leer_banda(salida, otra, 2014, 0, 1)


def test_command_line_builds_the_requested_years(tmp_path, capsys):
    preparar(tmp_path / "bruto", imagen_con_interfaz())
    salida = tmp_path / "interfaz"
    codigo = main(
        ["interfaz", "construir", "--bruto", str(tmp_path / "bruto"), "--salida", str(salida),
         "--anios", "2015", "2017"]
    )
    assert codigo == 0
    texto = capsys.readouterr().out
    assert "2015:" in texto and "2017:" in texto and "2 years written" in texto
    assert sorted(p.name for p in salida.glob("*.u8")) == [
        "banda_2015.u8", "banda_2017.u8", "dist40_2015.u8", "dist40_2017.u8"
    ]
