import hashlib
import io
import json
import zipfile

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from nadir import banco
from nadir.capas import fuego
from nadir.errors import SourceError
from nadir.estudio import validacion
from sinteticos import PASO, exportar

UTM = "EPSG:32719"


def _tif(ruta, datos, compresion=None):
    opciones = {"compress": compresion} if compresion else {}
    with rasterio.open(ruta, "w", driver="GTiff", width=datos.shape[1], height=datos.shape[0],
                       count=1, dtype="uint8", crs=UTM,
                       transform=from_origin(250000, 6340000, 20, 20), **opciones) as w:
        w.write(datos, 1)


def test_content_hash_ignores_compression(tmp_path):
    datos = np.arange(48, dtype=np.uint8).reshape(6, 8)
    _tif(tmp_path / "a.tif", datos)
    _tif(tmp_path / "b.tif", datos, "deflate")
    assert validacion.huella_contenido(tmp_path / "a.tif") == validacion.huella_contenido(
        tmp_path / "b.tif")
    _tif(tmp_path / "c.tif", datos + 1)
    assert validacion.huella_contenido(tmp_path / "c.tif") != validacion.huella_contenido(
        tmp_path / "a.tif")
    with pytest.raises(SourceError, match="content hash"):
        validacion.leer_clases(tmp_path / "a.tif")


def test_file_check(tmp_path):
    ruta = tmp_path / "x.zip"
    ruta.write_bytes(b"abc")
    sha = hashlib.sha256(b"abc").hexdigest()
    validacion.comprobar_archivo(ruta, 3, sha)
    with pytest.raises(SourceError, match="bytes"):
        validacion.comprobar_archivo(ruta, 4, sha)
    with pytest.raises(SourceError, match="sha256"):
        validacion.comprobar_archivo(ruta, 3, "0" * 64)
    with pytest.raises(SourceError, match="not found"):
        validacion.comprobar_archivo(tmp_path / "y.zip", 3, sha)


def _cuadrado(lon0, lat0, lado, **props):
    anillo = [[lon0, lat0], [lon0 + lado, lat0], [lon0 + lado, lat0 + lado],
              [lon0, lat0 + lado], [lon0, lat0]]
    return {"type": "Feature", "properties": props,
            "geometry": {"type": "Polygon", "coordinates": [anillo]}}


def test_municipalities_and_zone_mask(tmp_path):
    feats = [_cuadrado(-71.6, -33.1, 0.01, CUT_COM=c) for c in validacion.COMUNAS]
    ruta = tmp_path / "comunas.geojson"
    ruta.write_text(json.dumps({"type": "FeatureCollection", "features": feats}), "utf-8")
    datos = validacion.leer_comunas(ruta)
    ruta.write_text(json.dumps({"type": "FeatureCollection", "features": feats[:3]}), "utf-8")
    with pytest.raises(SourceError, match="municipalities"):
        validacion.leer_comunas(ruta)
    from pyproj import Transformer

    x, y = Transformer.from_crs("EPSG:4326", UTM, always_xy=True).transform(-71.6, -33.1)
    transform = from_origin(x - 200, y + 1300, 20, 20)
    mascara = validacion.mascara_zonas(datos, UTM, transform, (100, 100))
    area = mascara.sum() * 400
    assert 0.8e6 < area < 1.1e6  # a 0.01 degree square is about 0.93 km2 at 33 S
    with pytest.raises(SourceError, match="empty"):
        validacion.mascara_zonas({"features": []}, UTM, transform, (10, 10))


def test_strata_sampling_keeps_its_distance_and_order():
    pred = np.zeros((60, 60), dtype=bool)
    pred[:30] = True
    valido = np.ones_like(pred)
    q, n = validacion.muestrear_estratos(pred, valido, 5, 7, 10)
    q2, n2 = validacion.muestrear_estratos(pred, valido, 5, 7, 10)
    assert (q == q2).all() and (n == n2).all()
    assert pred.ravel()[q].all() and not pred.ravel()[n].any()
    for grupo in (q, n):
        filas, columnas = np.divmod(grupo, 60)
        for i in range(len(grupo)):
            for j in range(i + 1, len(grupo)):
                assert (abs(filas[i] - filas[j]) >= 10) or (abs(columnas[i] - columnas[j]) >= 10)
    with pytest.raises(SourceError, match="burned stratum"):
        validacion.muestrear_estratos(pred, valido, 50, 7, 10)
    with pytest.raises(SourceError, match="2-D"):
        validacion.muestrear_estratos(pred.ravel(), valido.ravel(), 5, 7)


def _paquete(ruta, duplicado_distinto=False):
    prod, v = validacion.EMSR715_PRODUCTO, "v2"
    quemado = {"features": [_cuadrado(-71.6, -33.1, 0.002)]}
    huella = {"features": [
        {**_cuadrado(-71.61, -33.11, 0.02), "properties": {"or_src_id": 2}},
        {**_cuadrado(-80.0, -40.0, 0.02), "properties": {"or_src_id": 1}},
    ]}
    interno = io.BytesIO()
    with zipfile.ZipFile(interno, "w") as z:
        z.writestr(f"{prod}_observedEventA_{v}.json", json.dumps(quemado))
        z.writestr(f"{prod}_imageFootprintA_{v}.json", json.dumps(huella))
    with zipfile.ZipFile(ruta, "w") as z:
        z.writestr(f"{prod}_{v}.zip", interno.getvalue())
        z.writestr(f"{prod}_{v}.zip", b"otro" if duplicado_distinto else interno.getvalue())
    return ruta


def test_emsr715_delineation_reads_nested_zips(tmp_path):
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # duplicate names inside the zip
        quemado, huella = validacion.delineacion_emsr715(_paquete(tmp_path / "p.zip"), UTM)
        assert len(quemado) == 1 and len(huella) == 1
        assert 3e4 < quemado[0].area < 5e4
        with pytest.raises(SourceError, match="different contents"):
            validacion.delineacion_emsr715(_paquete(tmp_path / "q.zip", True), UTM)


def test_fire_value_at_each_pixel_centre(tmp_path):
    alto, ancho = 40, 40
    mensual = np.zeros((13, alto, ancho), dtype=np.uint8)
    mensual[11] = np.arange(alto * ancho, dtype=np.uint32).reshape(alto, ancho) % 13
    anual = (mensual > 0).astype(np.uint8)
    carpeta = tmp_path / "bruto" / fuego.CARPETA
    banco.preparar(carpeta, fuego.FUENTE)
    exportar(carpeta / "d.zip", fuego.PREFIJO_ANUAL, anual,
             tuple(fuego.banda_anual(a) for a in fuego.ANIOS), lado=16)
    exportar(carpeta / "d.zip", fuego.PREFIJO_MENSUAL, mensual,
             tuple(fuego.banda_mensual(a) for a in fuego.ANIOS), lado=16)
    banco.registrar(carpeta, "d.zip", "2013-2025", "earthengine:FIRE", False)
    from pyproj import Transformer

    a_utm = Transformer.from_crs("EPSG:4326", UTM, always_xy=True)
    x0, y0 = a_utm.transform(-72.0 + 10 * PASO, -33.0 - 10 * PASO)
    transform = from_origin(x0, y0, 20, 20)
    valores = validacion.fuego_en_centros(tmp_path / "bruto", transform, UTM, (12, 12))
    a_geo = Transformer.from_crs(UTM, "EPSG:4326", always_xy=True)
    for f, c in ((0, 0), (5, 7), (11, 11)):
        lon, lat = a_geo.transform(x0 + (c + 0.5) * 20, y0 - (f + 0.5) * 20)
        fila = int(np.floor((-33.0 - lat) / PASO))
        col = int(np.floor((lon + 72.0) / PASO))
        assert valores[f, c] == mensual[11, fila, col]


def test_agreement_both_ways():
    clases = np.array([[0, 1, 3, 4], [5, 6, 7, 7]], dtype=np.uint8)
    en = np.ones_like(clases, dtype=bool)
    q = np.array([[True, False, False, True], [True, True, True, False]])
    r = validacion._concordancia(clases, en, q, 0.04)
    assert r["dominio_ha"] == 0.3
    assert r["clase0_en_comunas_ha"] == 0.0
    assert r["a_por_clase"]["7 High severity"] == {"ha": 0.1, "fuego_quemado_ha": 0.0, "pct": 50.0}
    assert r["a_por_clase"]["2 Enhanced regrowth, low"]["pct"] is None
    assert r["b_del_area_fuego"]["clases 5 a 7"]["pct"] == 60.0
    assert r["b_del_area_fuego"]["clase 0 (sin dato)"]["pct"] == 20.0


def test_perfect_prediction_has_full_accuracy():
    rng = np.random.default_rng(1)
    ref = np.zeros((420, 420), dtype=bool)  # room for 300 points 10 pixels apart per stratum
    ref[:210] = True
    dominio = rng.random(ref.shape) > 0.1
    r = validacion._exactitud(ref, ref, dominio, 0.04)
    assert r["exactitud_global_pct"] == 100.0
    assert r["comision_pct"] == 0.0 and r["omision_pct"] == 0.0
    assert r["matriz_muestra"] == {"fuego_q_ref_q": 300, "fuego_q_ref_n": 0,
                                   "fuego_n_ref_q": 0, "fuego_n_ref_n": 300}
    assert r["area_quemada_ajustada_ha"] == round((ref & dominio).sum() * 0.04, 1)
