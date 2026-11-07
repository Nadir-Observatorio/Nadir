import math

import numpy as np
import pytest

from nadir.area import area_filas_m2, area_franja_m2
from nadir.errors import GridError

PASO = 0.0002694945852358564


def test_whole_ellipsoid_matches_the_wgs84_surface():
    # Surface of the WGS84 ellipsoid: 510,065,621.72 km².
    assert area_franja_m2(-90, 90, 360) / 1e6 == pytest.approx(510_065_621.72, abs=0.05)


def test_one_degree_cell_at_the_equator():
    # 1° x 1° at the equator on WGS84: about 12,308.8 km².
    assert area_franja_m2(0, 1, 1) / 1e6 == pytest.approx(12_308.8, abs=0.5)


def test_pixel_near_valparaiso_is_about_754_square_metres():
    area = area_filas_m2(-33.0, PASO, 1)[0]
    assert 750 < area < 760


def test_rows_add_up_to_the_strip():
    filas = area_filas_m2(-33.0, PASO, 1000)
    franja = area_franja_m2(-33.0, -33.0 - 1000 * PASO, PASO)
    assert float(np.sum(filas)) == pytest.approx(franja, rel=1e-9)


def test_pixels_shrink_towards_the_pole_and_are_symmetric():
    norte = area_filas_m2(-30.0, 1.0, 20)
    assert np.all(np.diff(norte) < 0)
    assert area_franja_m2(10, 11, 1) == pytest.approx(area_franja_m2(-11, -10, 1), rel=1e-12)


def test_close_to_a_sphere_of_mean_radius():
    radio = 6_371_008.8
    esfera = (
        radio**2 * math.radians(1) * (math.sin(math.radians(-32)) - math.sin(math.radians(-33)))
    )
    assert area_franja_m2(-33, -32, 1) == pytest.approx(abs(esfera), rel=0.005)


@pytest.mark.parametrize(
    ("norte", "paso", "filas"),
    [(-33.0, 0.0, 1), (-33.0, -1.0, 1), (-33.0, 1.0, 0), (-89.5, 1.0, 2), (float("nan"), 1.0, 1)],
)
def test_bad_grids_are_refused(norte, paso, filas):
    with pytest.raises(GridError):
        area_filas_m2(norte, paso, filas)


@pytest.mark.parametrize("ancho", [0, -1, 361, float("inf")])
def test_bad_widths_are_refused(ancho):
    with pytest.raises(GridError):
        area_franja_m2(0, 1, ancho)


def test_distance_on_the_mean_sphere():
    from nadir.area import distancia_km

    # One degree of latitude on the mean sphere: 111.195 km.
    assert float(distancia_km(0, 0, 1, 0)) == pytest.approx(111.195, abs=0.001)
    ida = distancia_km([-33.0, 0.0], [-71.0, 0.0], [-33.5, 0.0], [-70.5, 0.0])
    vuelta = distancia_km([-33.5, 0.0], [-70.5, 0.0], [-33.0, 0.0], [-71.0, 0.0])
    np.testing.assert_allclose(ida, vuelta)
    assert ida[1] == 0
    with pytest.raises(GridError):
        distancia_km(91, 0, 0, 0)
