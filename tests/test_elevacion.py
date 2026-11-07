import hashlib

import numpy as np
import pytest

from nadir import banco, exportacion
from nadir.__main__ import main
from nadir.capas import cobertura, elevacion
from nadir.errors import SourceError
from sinteticos import exportar


def test_tile_names_follow_the_south_west_corner():
    assert elevacion.nombre(-36, -73) == "Copernicus_DSM_COG_10_S36_00_W073_00_DEM"
    assert elevacion.nombre(5, 12) == "Copernicus_DSM_COG_10_N05_00_E012_00_DEM"
    assert elevacion.url(elevacion.nombre(-36, -73)) == (
        "https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_S36_00_W073_00_DEM/"
        "Copernicus_DSM_COG_10_S36_00_W073_00_DEM.tif"
    )
    with pytest.raises(SourceError):
        elevacion.nombre(-91, 0)


def test_tiles_are_those_with_data(tmp_path):
    paso = 0.25
    imagen = np.full((1, 8, 8), 255, dtype=np.uint8)
    imagen[0, 0, 0] = 3  # -35.0 to -35.25, -73.0 to -72.75: tile S36 W073
    imagen[0, 5, 6] = 3  # -36.25 to -36.5, -71.5 to -71.25: tile S37 W072
    imagen[0, 7, 7] = 0  # MapBiomas no data still counts as inside the export
    ruta = exportar(
        tmp_path / "e.zip", "p", imagen, ("b",), lado=8, paso=paso, oeste=-73.0, norte=-35.0
    )
    with exportacion.abrir([ruta], "p") as mosaico:
        assert elevacion.teselas(mosaico, "b") == [
            "Copernicus_DSM_COG_10_S36_00_W073_00_DEM",
            "Copernicus_DSM_COG_10_S37_00_W072_00_DEM",
        ]


@pytest.fixture
def remoto(monkeypatch):
    estados = {}
    llamadas = []

    def bajar(url, carpeta, nombre, periodo, verificada):
        llamadas.append(url)
        datos = url.encode()
        (carpeta / nombre).write_bytes(datos)
        archivo = banco.Archivo(
            nombre=nombre,
            periodo=periodo,
            url=url,
            url_verificada=verificada,
            fecha_descarga="2026-11-05T10:00:00-03:00",
            bytes=len(datos),
            sha256=hashlib.sha256(datos).hexdigest(),
        )
        banco.agregar(carpeta, archivo)
        return archivo

    monkeypatch.setattr(banco, "estado_remoto", lambda u: (estados.get(u, 200), None))
    monkeypatch.setattr(banco, "descargar", bajar)
    return estados, llamadas


def test_download_skips_recorded_and_missing_tiles(tmp_path, remoto):
    estados, llamadas = remoto
    a, b, c = (elevacion.nombre(-36, x) for x in (-73, -72, -74))
    estados[elevacion.url(c)] = 404
    nuevos, ausentes = elevacion.descargar(tmp_path, [a, b, c])
    assert [n.nombre for n in nuevos] == [f"{a}.tif", f"{b}.tif"]
    assert all(n.url_verificada and n.periodo == "2010-2015" for n in nuevos)
    assert ausentes == [c]
    otra_vez, _ = elevacion.descargar(tmp_path, [a, b, c])
    assert otra_vez == []
    assert len(llamadas) == 2
    fuente, _ = banco.leer(tmp_path / elevacion.CARPETA)
    assert fuente.licencia == "Copernicus-DEM-GLO-30"


def test_unexpected_answer_stops(tmp_path, remoto):
    estados, _ = remoto
    a = elevacion.nombre(-36, -73)
    estados[elevacion.url(a)] = 500
    with pytest.raises(SourceError, match="HTTP 500"):
        elevacion.descargar(tmp_path, [a])


def test_cli_downloads_what_the_land_cover_needs(tmp_path, remoto, capsys):
    carpeta = tmp_path / cobertura.CARPETA
    banco.preparar(carpeta, cobertura.FUENTE)
    imagen = np.full((13, 4, 4), 66, dtype=np.uint8)
    bandas = tuple(cobertura.banda(a) for a in cobertura.ANIOS_ESTUDIO)
    exportar(carpeta / "c.zip", cobertura.PREFIJO, imagen, bandas, lado=4)
    banco.registrar(carpeta, "c.zip", "2012-2024", "earthengine:x", False)
    assert main(["elevacion", "descargar", "--bruto", str(tmp_path)]) == 0
    salida = capsys.readouterr().out
    assert "1 tiles cover the land cover export" in salida
    assert "Copernicus_DSM_COG_10_S34_00_W072_00_DEM.tif:" in salida
    assert "1 downloaded now; 1 tiles in the bank" in salida


ARCO = elevacion.ARCO


def _tesela(carpeta, lat_sur, lon_oeste, valores):
    """Write a GLO-30-like tile (pixel centres on whole arc seconds) and record it."""
    import rasterio
    from rasterio.transform import from_origin

    nombre = f"{elevacion.nombre(lat_sur, lon_oeste)}.tif"
    transformada = from_origin(lon_oeste - 0.5 / ARCO, lat_sur + 1 + 0.5 / ARCO, 1 / ARCO, 1 / ARCO)
    with rasterio.open(
        carpeta / nombre,
        "w",
        driver="GTiff",
        width=ARCO,
        height=ARCO,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transformada,
        compress="deflate",
    ) as destino:
        destino.write(valores.astype("float32"), 1)
    banco.registrar(carpeta, nombre, "2010-2015", elevacion.url(nombre[:-4]), True)


def test_horn_slope_of_flat_and_tilted_planes():
    plano = np.zeros((5, 6))
    assert np.allclose(elevacion.pendiente_horn(plano, 126000), 0)
    columnas = np.tile(np.arange(6, dtype=float) * 10, (5, 1))
    pendiente = elevacion.pendiente_horn(columnas, 126000)
    lat = -(126000 + 1) / ARCO
    dx = 30.92 * np.cos(np.radians(lat))  # about 25.3 m east-west at 35° S
    assert pendiente.shape == (3, 4)
    assert np.allclose(pendiente, np.degrees(np.arctan(10 / dx)), atol=0.05)
    con_hueco = columnas.copy()
    con_hueco[2, 2] = np.nan
    assert np.isnan(elevacion.pendiente_horn(con_hueco, 126000)[0, 0])


def test_slope_bands_follow_the_cuts():
    valores = np.array([0, 9.99, 10.0, 24.99, 25.0, 80, np.nan])
    assert elevacion.tramo(valores).tolist() == [0, 0, 1, 1, 2, 2, 255]


def test_slope_and_altitude_on_the_land_cover_grid(tmp_path):
    bruto = tmp_path / "bruto"
    carpeta = bruto / elevacion.CARPETA
    banco.preparar(carpeta, elevacion.FUENTE)
    columnas = np.arange(ARCO, dtype=np.float64)
    # Two tiles side by side rising 1 m per column from 73° W: one tilt across their edge.
    _tesela(carpeta, -36, -73, np.tile(columnas, (ARCO, 1)))
    _tesela(carpeta, -36, -72, np.tile(columnas + ARCO, (ARCO, 1)))
    paso = 0.0002694945852358564
    imagen = np.full((1, 6, 10), 66, dtype=np.uint8)
    imagen[0, 0, 0] = 255  # outside the export
    oeste = -72.0 - 5 * paso  # five pixels on each side of the tile edge
    ruta = exportar(
        tmp_path / "e.zip", "p", imagen, ("b",), lado=8, paso=paso, oeste=oeste, norte=-35.5
    )
    with exportacion.abrir([ruta], "p") as mosaico:
        registro = elevacion.construir_pendiente(bruto, mosaico, "b", tmp_path / "pend", alto=4)
        g = mosaico.grilla
        tramos = elevacion.leer_tramos(tmp_path / "pend", g, 0, 6)
    assert tramos[0, 0] == elevacion.SIN_TRAMO
    assert np.all(tramos.ravel()[1:] == 0)  # atan(1 / 25.3) is about 2.3°
    assert registro["pixeles_por_tramo"] == [59, 0, 0]
    assert registro["pixeles_dentro_sin_dem"] == 0
    altitud = np.fromfile(tmp_path / "pend" / "altitud_m.i16", dtype="<i2").reshape(6, 10)
    columnas_dem = np.rint(g.longitudes(0, 10) * ARCO)
    assert altitud[0, 0] == elevacion.SIN_ALTITUD
    assert altitud[3].tolist() == (columnas_dem + 73 * ARCO).astype(int).tolist()
    assert len(registro["teselas_sha256"]) == 2
    with exportacion.abrir([ruta], "p") as mosaico, pytest.raises(SourceError, match="overwrite"):
        elevacion.construir_pendiente(bruto, mosaico, "b", tmp_path / "pend")


def test_pixels_without_elevation_are_counted(tmp_path):
    bruto = tmp_path / "bruto"
    carpeta = bruto / elevacion.CARPETA
    banco.preparar(carpeta, elevacion.FUENTE)
    _tesela(carpeta, -36, -72, np.zeros((ARCO, ARCO)))
    paso = 0.0002694945852358564
    imagen = np.full((1, 4, 6), 66, dtype=np.uint8)
    ruta = exportar(
        tmp_path / "e.zip",
        "p",
        imagen,
        ("b",),
        lado=8,
        paso=paso,
        oeste=-72.0 - 3 * paso,
        norte=-35.5,
    )
    with exportacion.abrir([ruta], "p") as mosaico:
        registro = elevacion.construir_pendiente(bruto, mosaico, "b", tmp_path / "pend")
        tramos = elevacion.leer_tramos(tmp_path / "pend", mosaico.grilla, 0, 4)
    # West of -72° there is no tile; the column on the edge lacks its western neighbour too.
    assert np.all(tramos[:, :3] == elevacion.SIN_TRAMO)
    assert np.all(tramos[:, 4:] == 0)
    assert registro["pixeles_dentro_sin_dem"] == int(np.sum(tramos == elevacion.SIN_TRAMO))
    otra = exportacion.Grilla(0.0, 0.0, paso, 6, 4)
    with pytest.raises(SourceError, match="another grid"):
        elevacion.leer_tramos(tmp_path / "pend", otra, 0, 4)


def test_each_pixel_takes_the_slope_of_the_elevation_pixel_holding_its_centre(tmp_path):
    bruto = tmp_path / "bruto"
    carpeta = bruto / elevacion.CARPETA
    banco.preparar(carpeta, elevacion.FUENTE)
    paso = 0.0002694945852358564
    norte = -35.5
    g = exportacion.Grilla(-72.5, norte, paso, 5, 12)
    filas_dem = np.rint(-g.latitudes(0, 12) * ARCO).astype(int)
    escalon = int(filas_dem[6])  # a 100 m step: steep only on the two DEM rows beside it
    local = escalon - 35 * ARCO
    valores = np.zeros((ARCO, ARCO))
    valores[local:, :] = 100.0
    _tesela(carpeta, -36, -73, valores)
    imagen = np.full((1, 12, 5), 66, dtype=np.uint8)
    ruta = exportar(
        tmp_path / "e.zip", "p", imagen, ("b",), lado=8, paso=paso, oeste=-72.5, norte=norte
    )
    with exportacion.abrir([ruta], "p") as mosaico:
        elevacion.construir_pendiente(bruto, mosaico, "b", tmp_path / "pend", alto=5)
        tramos = elevacion.leer_tramos(tmp_path / "pend", mosaico.grilla, 0, 12)
    empinadas = np.isin(filas_dem, [escalon - 1, escalon])
    assert np.all(tramos[empinadas] == 2)
    assert np.all(tramos[~empinadas] == 0)
    assert empinadas.sum() >= 2


def test_cli_builds_slope_on_the_land_cover_grid(tmp_path, capsys):
    carpeta = tmp_path / elevacion.CARPETA
    banco.preparar(carpeta, elevacion.FUENTE)
    _tesela(carpeta, -34, -72, np.zeros((ARCO, ARCO)))
    cob = tmp_path / cobertura.CARPETA
    banco.preparar(cob, cobertura.FUENTE)
    imagen = np.full((13, 4, 4), 66, dtype=np.uint8)
    bandas = tuple(cobertura.banda(a) for a in cobertura.ANIOS_ESTUDIO)
    exportar(cob / "c.zip", cobertura.PREFIJO, imagen, bandas, lado=4, oeste=-71.5, norte=-33.5)
    banco.registrar(cob, "c.zip", "2012-2024", "earthengine:x", False)
    salida = tmp_path / "pend"
    argumentos = ["elevacion", "pendiente", "--bruto", str(tmp_path), "--salida", str(salida)]
    assert main(argumentos) == 0
    texto = capsys.readouterr().out
    assert "slope below 10 degrees: 16 pixels (100.0 %)" in texto
    assert "without elevation: 0" in texto
    assert main(argumentos) == 1
    assert "refusing to overwrite" in capsys.readouterr().err
