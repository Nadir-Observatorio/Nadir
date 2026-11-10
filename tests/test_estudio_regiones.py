import json

import h3.api.numpy_int as h3n
import numpy as np
import pandas as pd
import pytest

from nadir.__main__ import main
from nadir.errors import SourceError
from nadir.estudio import categorias, mapas, regiones
from test_estudio_analisis import _escribir

pytest.importorskip("shapely")


def _limites(carpeta, ruta, corte_extra=0.0):
    k3 = categorias.cargar(carpeta / "k3.parquet")
    lat = np.array([h3n.cell_to_latlng(int(c))[0] for c in np.unique(k3.celda)])
    lon = np.array([h3n.cell_to_latlng(int(c))[1] for c in np.unique(k3.celda)])
    corte = float(np.median(lon)) + corte_extra
    s, n = float(lat.min()) - 0.1, float(lat.max()) + 0.1
    o = float(lon.min()) - 0.1

    def caja(x0, x1):
        return {"type": "Polygon", "coordinates": [[[x0, s], [x1, s], [x1, n], [x0, n], [x0, s]]]}

    datos = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"NAME": "Oeste"}, "geometry": caja(o, corte)},
        {"type": "Feature", "properties": {"NAME": "Este"},
         "geometry": caja(corte, float(np.quantile(lon, 0.9)))},
    ]}
    ruta.write_text(json.dumps(datos), encoding="utf-8")
    return corte


def test_regions_add_up_to_h1_and_follow_the_cell_centre(tmp_path):
    carpeta = _escribir(tmp_path / "t")
    ruta = tmp_path / "r.geojson"
    corte = _limites(carpeta, ruta)
    tabla, meta = regiones.por_region(carpeta, ruta)
    b = mapas.base(carpeta)
    assert tabla.aporte_pp.sum() == pytest.approx(b.h1_pp, abs=1e-9)
    assert meta["h1_pp"] == round(float(b.h1_pp), 4)
    assert set(tabla.region) == {"Oeste", "Este", regiones.SIN_REGION}
    k3 = b.k3[b.k3.grupo.to_numpy() == 1]
    lon = np.array([h3n.cell_to_latlng(int(c))[1] for c in k3.celda])
    oeste = float(k3.area_m2[lon < corte].sum()) / 1e4
    fila = tabla.set_index("region").loc["Oeste"]
    assert fila.quemado_ha == pytest.approx(oeste)
    assert meta["celdas_r8_en_borde_ha"] > 0


def test_overlapping_regions_stop(tmp_path):
    carpeta = _escribir(tmp_path / "t")
    ruta = tmp_path / "r.geojson"
    _limites(carpeta, ruta)
    datos = json.loads(ruta.read_text(encoding="utf-8"))
    datos["features"][1]["geometry"] = datos["features"][0]["geometry"]
    datos["features"][1]["properties"]["NAME"] = "Copia"
    ruta.write_text(json.dumps(datos), encoding="utf-8")
    with pytest.raises(SourceError, match="overlap"):
        regiones.por_region(carpeta, ruta)


def test_command_line_writes_csv_and_json(tmp_path, capsys):
    carpeta = _escribir(tmp_path / "t")
    ruta = tmp_path / "r.geojson"
    _limites(carpeta, ruta)
    salida = tmp_path / "s"
    assert main(["estudio", "regiones", "--tablas", str(carpeta), "--limites", str(ruta),
                 "--salida", str(salida)]) == 0
    assert "crossing a boundary" in capsys.readouterr().out
    assert len(pd.read_csv(salida / "regiones.csv")) == 3
    for archivo in salida.iterdir():
        assert b"\r" not in archivo.read_bytes()
    with pytest.raises(SourceError, match="overwrite"):
        regiones.escribir(*regiones.por_region(carpeta, ruta), salida)
