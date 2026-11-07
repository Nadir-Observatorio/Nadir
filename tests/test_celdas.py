import json

import h3.api.numpy_int as h3n
import numpy as np
import pytest

from nadir import celdas, exportacion, grilla
from nadir.errors import GridError, LayerError, ProvenanceError
from sinteticos import exportar

BANDAS = ("v_2023", "v_2024")
PASO = 0.002  # coarse pixels, so a small image spans several resolution-8 cells


@pytest.fixture
def mosaico(tmp_path):
    rng = np.random.default_rng(3)
    imagen = rng.integers(0, 4, size=(2, 30, 40)).astype(np.uint8)
    imagen[:, :3, :] = 255  # outside the area
    imagen[:, 10, 5] = 255
    ruta = exportar(tmp_path / "e.zip", "pre", imagen, BANDAS, lado=16, paso=PASO)
    with exportacion.abrir([ruta], "pre") as abierto:
        yield abierto, imagen


def _centro(g, fila, columna, resolucion):
    lat = float(g.latitudes(fila, 1)[0])
    lon = float(g.longitudes(columna, 1)[0])
    return h3n.latlng_to_cell(lat, lon, resolucion)


def test_every_pixel_points_to_the_cell_of_its_centre(tmp_path, mosaico):
    abierto, imagen = mosaico
    mapa = celdas.construir(abierto, "v_2024", 8, tmp_path / "m8", alto=7)
    assert np.all(np.diff(mapa.celdas) > 0)
    indices = mapa.indices(0, 30)
    for fila in range(30):
        for columna in range(40):
            if imagen[1, fila, columna] == 255:
                assert indices[fila, columna] == celdas.SIN_CELDA
            else:
                esperado = _centro(abierto.grilla, fila, columna, 8)
                assert mapa.celdas[indices[fila, columna]] == esperado


def test_table_matches_a_pixel_by_pixel_sum(tmp_path, mosaico):
    abierto, imagen = mosaico
    mapa = celdas.construir(abierto, "v_2024", 7, tmp_path / "m7")
    lista, tabla = celdas.tabular(mapa, abierto, "v_2023", alto=4)
    areas = abierto.grilla.area_filas_m2(0, 30)
    esperado: dict[tuple[int, int], float] = {}
    for fila in range(30):
        for columna in range(40):
            if imagen[1, fila, columna] == 255:
                continue
            clave = (_centro(abierto.grilla, fila, columna, 7), int(imagen[0, fila, columna]))
            esperado[clave] = esperado.get(clave, 0.0) + areas[fila]
    obtenido = {
        (int(lista[i]), int(v)): float(tabla[i, v]) for i, v in zip(*np.nonzero(tabla), strict=True)
    }
    assert obtenido.keys() == esperado.keys()
    for clave, valor in esperado.items():
        assert obtenido[clave] == pytest.approx(valor, rel=1e-9)


def test_centre_tables_agree_with_grilla_agregar(tmp_path, mosaico):
    abierto, imagen = mosaico
    mapa = celdas.construir(abierto, "v_2024", 7, tmp_path / "m7")
    lista, tabla = celdas.tabular(mapa, abierto, "v_2024")
    filas, columnas = np.nonzero(imagen[1] != 255)
    g = abierto.grilla
    valores = (imagen[1][filas, columnas] == 2).astype(float)
    km2 = g.area_filas_m2(0, 30)[filas] / 1e6
    referencia = grilla.agregar(
        g.latitudes(0, 30)[filas], g.longitudes(0, 40)[columnas], valores, km2, 7, "fraccion"
    )
    fraccion = tabla[:, 2] / tabla[:, :255].sum(axis=1)
    nombres = [h3n.int_to_str(int(c)) for c in lista]
    assert sorted(nombres) == list(referencia)
    for nombre, valor in zip(nombres, fraccion, strict=True):
        assert referencia[nombre].valor == pytest.approx(valor, rel=1e-9)


def test_coarser_tables_add_up_children(tmp_path, mosaico):
    abierto, _ = mosaico
    mapa = celdas.construir(abierto, "v_2024", 8, tmp_path / "m8")
    finas, tabla8 = celdas.tabular(mapa, abierto, "v_2023")
    for resolucion in (7, 5, 3):
        gruesas, tabla = celdas.tabular(mapa, abierto, "v_2023", resolucion)
        padres = np.array([h3n.cell_to_parent(int(c), resolucion) for c in finas])
        assert sorted(set(padres.tolist())) == gruesas.tolist()
        for i, celda in enumerate(gruesas):
            np.testing.assert_allclose(tabla[i], tabla8[padres == celda].sum(axis=0), rtol=1e-9)
    with pytest.raises(GridError, match="finer than the map"):
        celdas.tabular(mapa, abierto, "v_2023", 9)
    with pytest.raises(GridError, match="start at 0"):
        mapa.padres(-1)


def test_map_is_saved_checked_and_never_overwritten(tmp_path, mosaico):
    abierto, _ = mosaico
    carpeta = tmp_path / "m8"
    construido = celdas.construir(abierto, "v_2024", 8, carpeta)
    registro = json.loads((carpeta / "mapa.json").read_text(encoding="utf-8"))
    assert registro["resolucion"] == 8
    assert registro["celdas"] == construido.celdas.size
    abierto_de_nuevo = celdas.abrir(carpeta, abierto.grilla)
    np.testing.assert_array_equal(abierto_de_nuevo.celdas, construido.celdas)
    assert abierto_de_nuevo.resolucion == 8
    with pytest.raises(LayerError, match="refusing to overwrite"):
        celdas.construir(abierto, "v_2024", 8, carpeta)
    otra = exportacion.Grilla(0.0, 0.0, PASO, 40, 30)
    with pytest.raises(ProvenanceError, match="another grid"):
        celdas.abrir(carpeta, otra)
    indices = carpeta / "indices.u32"
    datos = bytearray(indices.read_bytes())
    datos[0] ^= 0xFF
    indices.write_bytes(bytes(datos))
    with pytest.raises(ProvenanceError, match="sha256"):
        celdas.abrir(carpeta, abierto.grilla)
    with pytest.raises(ProvenanceError, match="no cell map"):
        celdas.abrir(tmp_path / "nada", abierto.grilla)


def test_bad_requests_are_refused(tmp_path, mosaico):
    abierto, _ = mosaico
    mapa = celdas.construir(abierto, "v_2024", 8, tmp_path / "m8")
    with pytest.raises(GridError, match="outside the map"):
        mapa.indices(29, 2)
    with pytest.raises(GridError):
        celdas.construir(abierto, "v_2024", 9, tmp_path / "m9")
    otra = exportar(tmp_path / "o.zip", "pre", np.zeros((2, 4, 4), dtype=np.uint8), BANDAS, lado=4)
    with exportacion.abrir([otra], "pre") as distinto, pytest.raises(GridError, match="different"):
        celdas.tabular(mapa, distinto, "v_2023")


def test_a_truncated_index_file_is_refused(tmp_path, mosaico):
    abierto, _ = mosaico
    carpeta = tmp_path / "m8"
    mapa = celdas.construir(abierto, "v_2024", 8, carpeta)
    indices = carpeta / "indices.u32"
    indices.write_bytes(indices.read_bytes()[:-4])
    with pytest.raises(ProvenanceError, match="shorter"):
        mapa.indices(29, 1)
