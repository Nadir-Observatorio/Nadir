import zipfile
from dataclasses import replace

import numpy as np
import pytest

from nadir import exportacion
from nadir.errors import GridError, SourceError
from sinteticos import NORTE, OESTE, PASO, exportar, tif

BANDAS = ("b_2023", "b_2024")


def imagen(alto=7, ancho=10):
    datos = np.arange(2 * alto * ancho, dtype=np.int64).reshape(2, alto, ancho) % 250
    return datos.astype(np.uint8)


def test_tiles_rebuild_the_image(tmp_path):
    original = imagen()
    ruta = exportar(tmp_path / "a.zip", "pre", original, BANDAS, lado=4)
    with exportacion.abrir([ruta], "pre") as mosaico:
        assert len(mosaico.teselas) == 6
        assert mosaico.grilla == exportacion.Grilla(OESTE, NORTE, PASO, 10, 7)
        assert mosaico.bandas == BANDAS
        assert mosaico.sin_dato == 255
        for i, banda in enumerate(BANDAS):
            np.testing.assert_array_equal(mosaico.leer(banda, 0, 7), original[i])
            np.testing.assert_array_equal(mosaico.leer(banda, 3, 3), original[i, 3:6])
            franjas = list(mosaico.franjas(banda, 3))
            assert [f for f, _ in franjas] == [0, 3, 6]
            np.testing.assert_array_equal(np.vstack([d for _, d in franjas]), original[i])


def test_tiles_can_be_split_across_zips(tmp_path):
    original = imagen()
    primera = exportar(tmp_path / "1.zip", "pre", original[:, :4], BANDAS, lado=4)
    segunda = exportar(
        tmp_path / "2.zip", "pre", original[:, 4:], BANDAS, lado=4, norte=NORTE - 4 * PASO
    )
    # Second zip: rename its tiles to their row offset in the whole image.
    renombrada = tmp_path / "2b.zip"
    with zipfile.ZipFile(segunda) as origen, zipfile.ZipFile(renombrada, "w") as destino:
        for nombre in origen.namelist():
            _, fila, _ = nombre[:-4].rsplit("-", 2)
            nuevo = nombre.replace(f"-{fila}-", f"-{int(fila) + 4:010d}-")
            destino.writestr(nuevo, origen.read(nombre))
    with exportacion.abrir([primera, renombrada], "pre") as mosaico:
        np.testing.assert_array_equal(mosaico.leer("b_2024", 0, 7), original[1])


def test_other_prefixes_and_files_are_ignored(tmp_path):
    ruta = exportar(tmp_path / "a.zip", "pre", imagen(), BANDAS, lado=4)
    exportar(ruta, "otro", imagen(), BANDAS, lado=4)
    with zipfile.ZipFile(ruta, "a") as archivo:
        archivo.writestr("exportacion_ee/notas.txt", "x")
    with exportacion.abrir([ruta], "pre") as mosaico:
        assert len(mosaico.teselas) == 6


def test_area_rows_follow_the_grid(tmp_path):
    ruta = exportar(tmp_path / "a.zip", "pre", imagen(), BANDAS, lado=4)
    with exportacion.abrir([ruta], "pre") as mosaico:
        areas = mosaico.grilla.area_filas_m2(0, 7)
        assert areas.shape == (7,)
        assert 750 < areas[0] < 760
        lat = mosaico.grilla.latitudes(0, 2)
        assert lat[0] == pytest.approx(NORTE - PASO / 2)
        lon = mosaico.grilla.longitudes(9, 1)
        assert lon[0] == pytest.approx(OESTE + 9.5 * PASO)


def test_no_tiles_is_an_error(tmp_path):
    ruta = exportar(tmp_path / "a.zip", "otro", imagen(), BANDAS, lado=4)
    with pytest.raises(SourceError, match="no tiles named pre"):
        exportacion.abrir([ruta], "pre")


def test_unreadable_zip_is_an_error(tmp_path):
    ruta = tmp_path / "roto.zip"
    ruta.write_bytes(b"no es un zip")
    with pytest.raises(SourceError, match="not a readable zip"):
        exportacion.abrir([ruta], "pre")


def test_missing_tile_is_refused(tmp_path):
    ruta = exportar(tmp_path / "a.zip", "pre", imagen(), BANDAS, lado=4)
    sin_una = tmp_path / "b.zip"
    with zipfile.ZipFile(ruta) as origen, zipfile.ZipFile(sin_una, "w") as destino:
        for nombre in origen.namelist():
            if not nombre.endswith("-0000000004-0000000004.tif"):
                destino.writestr(nombre, origen.read(nombre))
    with pytest.raises(SourceError, match="5 tiles, expected 6"):
        exportacion.abrir([sin_una], "pre")


def test_gap_between_tiles_is_refused(tmp_path):
    datos = imagen()
    ruta = tmp_path / "hueco.zip"
    with zipfile.ZipFile(ruta, "w") as archivo:
        archivo.writestr("pre-0000000000-0000000000.tif", tif(datos[:, :4], BANDAS))
        abajo = tif(datos[:, 4:], BANDAS, norte=NORTE - 5 * PASO)
        archivo.writestr("pre-0000000005-0000000000.tif", abajo)
    with pytest.raises(GridError, match="gap or overlap along the rows at 4"):
        exportacion.abrir([ruta], "pre")


def test_duplicate_tile_is_refused(tmp_path):
    ruta = exportar(tmp_path / "a.zip", "pre", imagen(), BANDAS, lado=4)
    otra = exportar(tmp_path / "b.zip", "pre", imagen()[:, :4, :4], BANDAS, lado=4)
    with pytest.raises(SourceError, match="more than once"):
        exportacion.abrir([ruta, otra], "pre")


def _con_tesela(tmp_path, nombre, contenido):
    ruta = exportar(tmp_path / "a.zip", "pre", imagen(), BANDAS, lado=4)
    nueva = tmp_path / "b.zip"
    with zipfile.ZipFile(ruta) as origen, zipfile.ZipFile(nueva, "w") as destino:
        for miembro in origen.namelist():
            destino.writestr(
                miembro, contenido if miembro.endswith(nombre) else origen.read(miembro)
            )
    return nueva


def test_tile_out_of_place_is_refused(tmp_path):
    pedazo = imagen()[:, 4:, 4:8]
    contenido = tif(pedazo, BANDAS, oeste=OESTE + 5 * PASO, norte=NORTE - 4 * PASO)
    ruta = _con_tesela(tmp_path, "-0000000004-0000000004.tif", contenido)
    with pytest.raises(GridError, match="not where its name places it"):
        exportacion.abrir([ruta], "pre")


@pytest.mark.parametrize(
    ("opciones", "mensaje"),
    [
        ({"crs": "EPSG:32719"}, "reference system"),
        ({"tipo": "uint16"}, "data type"),
        ({"sin_dato": 0}, "no-data value"),
        ({"paso": PASO * 2}, "cell size"),
    ],
)
def test_inconsistent_tile_is_refused(tmp_path, opciones, mensaje):
    pedazo = imagen()[:, 4:, 4:8]
    contenido = tif(pedazo, BANDAS, oeste=OESTE + 4 * PASO, norte=NORTE - 4 * PASO, **opciones)
    ruta = _con_tesela(tmp_path, "-0000000004-0000000004.tif", contenido)
    with pytest.raises((SourceError, GridError), match=mensaje):
        exportacion.abrir([ruta], "pre")


def test_unreadable_tile_is_refused(tmp_path):
    ruta = _con_tesela(tmp_path, "-0000000004-0000000004.tif", b"not a tiff")
    with pytest.raises(SourceError, match="cannot be read"):
        exportacion.abrir([ruta], "pre")


def test_other_band_names_are_refused(tmp_path):
    pedazo = imagen()[:, 4:, 4:8]
    contenido = tif(pedazo, ("x", "y"), oeste=OESTE + 4 * PASO, norte=NORTE - 4 * PASO)
    ruta = _con_tesela(tmp_path, "-0000000004-0000000004.tif", contenido)
    with pytest.raises(SourceError, match="bands"):
        exportacion.abrir([ruta], "pre")


def test_tile_without_no_data_is_refused(tmp_path):
    ruta = exportar(tmp_path / "a.zip", "pre", imagen(), BANDAS, lado=10, sin_dato=None)
    with pytest.raises(SourceError, match="no no-data value"):
        exportacion.abrir([ruta], "pre")


def test_bad_reads_are_refused(tmp_path):
    ruta = exportar(tmp_path / "a.zip", "pre", imagen(), BANDAS, lado=4)
    with exportacion.abrir([ruta], "pre") as mosaico:
        with pytest.raises(SourceError, match="no band"):
            mosaico.leer("b_1999", 0, 1)
        with pytest.raises(GridError, match="outside the image"):
            mosaico.leer("b_2023", 5, 3)
        with pytest.raises(GridError, match="strip height"):
            list(mosaico.franjas("b_2023", 0))


def test_offset_between_aligned_grids():
    base = exportacion.Grilla(-77.65243928444491, -15.414012297150046, PASO, 100, 100)
    otra = exportacion.Grilla(-74.84861761965107, -32.020538133968756, PASO, 10, 10)
    assert exportacion.desfase(base, otra) == (10404, 61621)


def test_misaligned_grids_are_refused():
    base = exportacion.Grilla(-72.0, -33.0, PASO, 10, 10)
    with pytest.raises(GridError, match="not aligned"):
        exportacion.desfase(base, exportacion.Grilla(-72.0 + PASO / 2, -33.0, PASO, 10, 10))
    with pytest.raises(GridError, match="cell sizes differ"):
        exportacion.desfase(base, exportacion.Grilla(-72.0, -33.0, PASO * 1.001, 10, 10))


def test_same_grid_tolerates_the_last_bit_of_the_cell_size():
    esquina = (-73.97383819597547, -32.020538133968756)
    cob = exportacion.Grilla(*esquina, 0.0002694945852358564, 15600, 28263)
    fue = exportacion.Grilla(*esquina, 0.00026949458523585647, 15600, 28263)
    assert cob != fue
    assert exportacion.misma_grilla(cob, fue)
    assert not exportacion.misma_grilla(cob, replace(fue, alto=28262))
    assert not exportacion.misma_grilla(cob, replace(fue, oeste=fue.oeste + fue.paso))
    assert not exportacion.misma_grilla(cob, replace(fue, oeste=fue.oeste + fue.paso / 3))
    assert not exportacion.misma_grilla(cob, replace(fue, paso=fue.paso * (1 + 1e-9)))
