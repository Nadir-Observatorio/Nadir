import math

import h3
import pytest

from nadir import grilla
from nadir.errors import GridError

VINA = (-33.03, -71.55)


def _puntos_en(celda, n):
    """n points inside one cell: its centre, repeated."""
    lat, lon = h3.cell_to_latlng(celda)
    return [lat] * n, [lon] * n


def test_only_observatory_resolutions_are_accepted():
    assert grilla.RESOLUCIONES == (5, 6, 7, 8)
    with pytest.raises(GridError, match="resolution 9"):
        grilla.celda(*VINA, 9)


def test_point_cell_and_parent_are_consistent():
    hijo = grilla.celda(*VINA, 7)
    assert h3.get_resolution(hijo) == 7
    assert grilla.padre(hijo, 6) == grilla.celda(*VINA, 6)
    with pytest.raises(GridError, match="finer"):
        grilla.padre(grilla.celda(*VINA, 6), 7)


def test_invalid_inputs_are_reported():
    with pytest.raises(GridError, match="out of range"):
        grilla.celda(-95.0, 0.0, 7)
    with pytest.raises(GridError, match="not a valid H3 cell"):
        grilla.padre("zzz", 6)
    with pytest.raises(GridError, match="not a valid H3 cell"):
        grilla.area_km2("zzz")


def test_area_of_a_resolution_7_cell_is_about_5_km2():
    assert 4.0 < grilla.area_km2(grilla.celda(*VINA, 7)) < 6.5


def test_fraction_rule_and_valid_area():
    celda = grilla.celda(*VINA, 7)
    area = grilla.area_km2(celda)
    lat, lon = _puntos_en(celda, 4)
    pixel = area / 8
    resultado = grilla.agregar(
        lat, lon, [1, 1, 0, 0], [pixel, pixel, pixel, pixel * 2], 7, "fraccion"
    )
    assert list(resultado) == [celda]
    assert math.isclose(resultado[celda].valor, 2 / 5)
    assert math.isclose(resultado[celda].fraccion_valida, 5 / 8)


def test_no_data_is_excluded_from_value_and_valid_area():
    celda = grilla.celda(*VINA, 7)
    lat, lon = _puntos_en(celda, 3)
    resultado = grilla.agregar(lat, lon, [1.0, float("nan"), 0.0], [0.1, 0.1, 0.1], 7, "fraccion")
    assert math.isclose(resultado[celda].valor, 0.5)
    assert math.isclose(resultado[celda].fraccion_valida, 0.2 / grilla.area_km2(celda))


def test_cell_without_valid_pixels_is_left_out():
    celda = grilla.celda(*VINA, 7)
    lat, lon = _puntos_en(celda, 2)
    assert grilla.agregar(lat, lon, [float("nan")] * 2, [0.1, 0.1], 7, "media") == {}


@pytest.mark.parametrize(
    ("regla", "esperado"),
    [("suma", 9.0), ("media", (2 * 1 + 3 * 1 + 4 * 2) / 4), ("moda", 4.0), ("conteo", 3.0)],
)
def test_other_rules(regla, esperado):
    celda = grilla.celda(*VINA, 7)
    lat, lon = _puntos_en(celda, 3)
    resultado = grilla.agregar(lat, lon, [2, 3, 4], [0.1, 0.1, 0.2], 7, regla)
    assert math.isclose(resultado[celda].valor, esperado)


def test_mode_tie_goes_to_smallest_value():
    celda = grilla.celda(*VINA, 7)
    lat, lon = _puntos_en(celda, 2)
    resultado = grilla.agregar(lat, lon, [9, 3], [0.1, 0.1], 7, "moda")
    assert resultado[celda].valor == 3.0


def test_points_in_two_cells_are_split_and_sorted():
    a = grilla.celda(*VINA, 7)
    b = grilla.celda(-38.74, -72.60, 7)
    la, oa = _puntos_en(a, 1)
    lb, ob = _puntos_en(b, 1)
    resultado = grilla.agregar(lb + la, ob + oa, [1, 0], [0.1, 0.1], 7, "fraccion")
    assert list(resultado) == sorted([a, b])
    assert resultado[a].valor == 0.0
    assert resultado[b].valor == 1.0


def test_valid_fraction_is_capped_at_one():
    celda = grilla.celda(*VINA, 7)
    lat, lon = _puntos_en(celda, 1)
    resultado = grilla.agregar(lat, lon, [1], [grilla.area_km2(celda) * 2], 7, "fraccion")
    assert resultado[celda].fraccion_valida == 1.0


def test_bad_requests_are_refused():
    celda = grilla.celda(*VINA, 7)
    lat, lon = _puntos_en(celda, 2)
    with pytest.raises(GridError, match="rule"):
        grilla.agregar(lat, lon, [1, 0], [0.1, 0.1], 7, "mediana")
    with pytest.raises(GridError, match="0 or 1"):
        grilla.agregar(lat, lon, [2, 0], [0.1, 0.1], 7, "fraccion")
    with pytest.raises(GridError, match="same length"):
        grilla.agregar(lat, lon, [1], [0.1, 0.1], 7, "fraccion")
    with pytest.raises(GridError, match="positive"):
        grilla.agregar(lat, lon, [1, 0], [0.1, 0.0], 7, "fraccion")
    with pytest.raises(GridError, match="resolution"):
        grilla.agregar(lat, lon, [1, 0], [0.1, 0.1], 4, "fraccion")


def test_mean_cell_diameter_doubles_the_mean_edge():
    diametro_medio_km = grilla.diametro_medio_km
    for resolucion in (3, 4, 5, 7):
        esperado = 2 * h3.average_hexagon_edge_length(resolucion, "km")
        assert diametro_medio_km(resolucion) == pytest.approx(esperado)
    assert diametro_medio_km(3) > diametro_medio_km(4) > diametro_medio_km(5)
    for mala in (-1, 16):
        with pytest.raises(GridError):
            diametro_medio_km(mala)
