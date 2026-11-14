import numpy as np
import pandas as pd
import pytest

from nadir.errors import SourceError
from nadir.estudio import contexto
from sinteticos import tif

openpyxl = pytest.importorskip("openpyxl")
pytest.importorskip("osmium")
pytest.importorskip("pyproj")

CAB = ["REGION", "PROVINCIA", "COMUNA", "NUMERO INCENDIOS ", *[None] * 15,
       "TOTAL SUPERFICIE AFECTADA"]


def _hoja(libro, nombre, bloques, nacional=None):
    hoja = libro.create_sheet(nombre)
    hoja.append(["RESUMEN NACIONAL DE OCURRENCIA POR COMUNAS"])
    hoja.append([f"PERIODO {nombre}"])
    hoja.append([nombre])
    hoja.append(CAB)
    total = 0.0
    for codigo, provincia, valores in bloques:
        for i, v in enumerate(valores):
            hoja.append([codigo if i == 0 else None, provincia if i == 0 else None, "X", 1,
                         *[None] * 15, v])
        hoja.append(["TOTAL", None, "TOTAL", 1, *[None] * 15, sum(valores)])
        total += sum(valores)
    hoja.append(["TOTAL NACIONAL", None, None, 1, *[None] * 15,
                 total if nacional is None else nacional])


def _libro(ruta, nacional=None):
    libro = openpyxl.Workbook()
    libro.remove(libro.active)
    for i, temporada in enumerate(contexto.TEMPORADAS):
        nombre = temporada.replace("-", " - ") if i % 2 else temporada
        bloques = [("V", "PETORCA", [1.0, 2.0]), (None, "ARAUCO", [10.0]),
                   ("XVI", "DIGUILLIN", [5.0]), ("RM", "CHACABUCO", [0.5])]
        _hoja(libro, nombre, bloques, nacional if i == 3 else None)
    libro.create_sheet("2011-2012")
    libro.save(ruta)
    return ruta


def test_conaf_regions_add_up_and_blank_codes_take_the_province(tmp_path):
    t = contexto.leer_conaf(_libro(tmp_path / "c.xlsx"))
    assert set(t.temporada) == set(contexto.TEMPORADAS)
    por = t.groupby("iso").ha.sum()
    n = len(contexto.TEMPORADAS)
    assert por["CL-VS"] == pytest.approx(3.0 * n)
    assert por["CL-BI"] == pytest.approx(15.0 * n)  # Biobío and Ñuble in one row
    assert por["CL-RM"] == pytest.approx(0.5 * n)


def test_conaf_totals_must_match(tmp_path):
    with pytest.raises(SourceError, match="add up"):
        contexto.leer_conaf(_libro(tmp_path / "c.xlsx", nacional=999.0))


def _cuadrado(x0, y0, x1, y1):
    return {"type": "Polygon", "coordinates": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]}


def test_interface_area_by_region(tmp_path):
    from pyproj import Transformer

    a = tmp_path / "a.tif"
    b = tmp_path / "b.tif"
    lado = 300.0
    oeste, norte = 300000.0, 6300000.0
    va = np.full((1, 20, 20), np.nan, dtype=np.float32)
    vb = np.full((1, 20, 20), np.nan, dtype=np.float32)
    va[0, :10, :10] = 7 / 12   # north-west quarter: interface
    vb[0, 10:, 10:] = 0.75     # south-east quarter: intermix
    va[0, 10:, :10] = 0.5      # not above the threshold
    for ruta, v in ((a, va), (b, vb)):
        ruta.write_bytes(tif(v, ("i",), oeste=oeste, norte=norte, paso=lado, crs="EPSG:32719",
                             sin_dato=np.nan, tipo="float32"))
    from shapely.geometry import box, mapping
    from shapely.ops import transform

    inv = Transformer.from_crs("EPSG:32719", "EPSG:4326", always_xy=True).transform

    def region(c0, f0, c1, f1):
        """Quarter of the raster by cell columns and rows, inset a quarter of a cell."""
        g = box(oeste + (c0 + 0.25) * lado, norte - (f1 - 0.25) * lado,
                oeste + (c1 - 0.25) * lado, norte - (f0 + 0.25) * lado).segmentize(lado / 4)
        return mapping(transform(inv, g))

    regiones = {"features": [
        {"properties": {"iso": "CL-VS"}, "geometry": region(0, 0, 10, 10)},
        {"properties": {"iso": "CL-NB"}, "geometry": region(10, 10, 20, 20)},
        {"properties": {"iso": "CL-ML"}, "geometry": region(0, 10, 10, 20)},
    ]}
    w = contexto.interfaz_por_region(a, b, regiones).set_index("iso").interfaz_ha
    celda = lado * lado / 1e4
    assert w["CL-VS"] == pytest.approx(100 * celda)
    assert w["CL-NB"] == pytest.approx(100 * celda)
    assert w["CL-ML"] == 0.0  # 0.5 is not above the threshold


def test_table_merges_nuble_and_marks_the_study_regions():
    conaf = pd.DataFrame({"temporada": ["2012-2013"] * 3, "iso": ["CL-VS", "CL-BI", "CL-AI"],
                          "ha": [10.0, 30.0, 60.0]})
    wui = pd.DataFrame({"iso": ["CL-VS", "CL-BI", "CL-NB", "CL-AI"],
                        "interfaz_ha": [5.0, 3.0, 2.0, 10.0]})
    t = contexto.tabla(conaf, wui).set_index("iso")
    assert list(t.index) == ["CL-VS", "CL-BI", "CL-AI"]  # north to south
    assert t.loc["CL-BI", "interfaz_ha"] == 5.0
    assert t.loc["CL-BI", "region"] == "Biobío y Ñuble"
    assert t.interfaz_pct_nacional.sum() == pytest.approx(100.0)
    assert t.loc["CL-AI", "afectada_pct_nacional"] == pytest.approx(60.0)
    assert t.estudio.tolist() == [True, True, False]


def test_osm_regions_are_read_from_boundary_relations(tmp_path):
    nodos = [(1, -71.0, -33.0), (2, -70.0, -33.0), (3, -70.0, -34.0), (4, -71.0, -34.0)]
    xml = ['<?xml version="1.0" encoding="UTF-8"?>', '<osm version="0.6">']
    for i, lon, lat in nodos:
        xml.append(f'<node id="{i}" version="1" lat="{lat}" lon="{lon}"/>')
    xml.append('<way id="10" version="1">' + "".join(f'<nd ref="{i}"/>' for i in (1, 2, 3, 4, 1))
               + "</way>")
    xml.append('<relation id="100" version="1"><member type="way" ref="10" role="outer"/>'
               '<tag k="type" v="boundary"/><tag k="boundary" v="administrative"/>'
               '<tag k="admin_level" v="4"/><tag k="ISO3166-2" v="CL-VS"/>'
               '<tag k="name" v="Región de Valparaíso"/></relation>')
    xml.append("</osm>")
    ruta = tmp_path / "x.osm"
    ruta.write_text("\n".join(xml), encoding="utf-8")
    g = contexto.regiones_osm(ruta)
    assert [f["properties"]["iso"] for f in g["features"]] == ["CL-VS"]
    assert g["features"][0]["geometry"]["type"] == "MultiPolygon"
