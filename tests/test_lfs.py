import hashlib
import json
import struct
import zipfile
from dataclasses import asdict
from datetime import date

import numpy as np
import pyarrow.parquet as pq
import pytest

from nadir import banco, exportacion
from nadir.capas import lfs
from nadir.errors import SourceError
from sinteticos import tif

PASO = 0.002
GRILLA = exportacion.Grilla(oeste=-72.0, norte=-33.0, paso=PASO, ancho=40, alto=30)
PASO_LFS = 0.0027  # coarser than the grid and not a multiple of it, like Landsat on 1"
FUENTE = banco.Fuente("Miranda et al.", "LFS", lfs.PAGINA, "CC-BY-4.0", "Miranda et al. (2022)")


def dbf(filas, campos=("FireID", "Season", "TotalArea", "OverlapIDs"), borrados=()):
    """Bytes of a dBASE III table with text fields of width 20."""
    ancho = 20
    cabecera = 32 + 32 * len(campos) + 1
    largo = 1 + ancho * len(campos)
    salida = bytearray(struct.pack("<BBBBIHH20x", 3, 126, 1, 1, len(filas), cabecera, largo))
    for nombre in campos:
        salida += nombre.encode("ascii").ljust(11, b"\x00") + b"C" + b"\x00" * 4
        salida += bytes([ancho, 0]) + b"\x00" * 14
    salida += b"\x0d"
    for i, fila in enumerate(filas):
        salida += b"*" if i in borrados else b" "
        for nombre in campos:
            salida += str(fila.get(nombre, "")).encode("latin-1")[:ancho].ljust(ancho)
    return bytes(salida + b"\x1a")


def cicatriz_tif(mascara, oeste, norte, paso=PASO_LFS):
    return tif(np.asarray(mascara, dtype=np.uint8)[None], ("b1",), oeste=oeste, norte=norte,
               paso=paso, sin_dato=None)


def agregar_fuego(zf, raiz, region, fid, umbral, fecha, mascara, oeste, norte, area_m2,
                  temporada=None, solapes=""):
    base = f"{raiz}/FireScar_{region}_ID{fid}_u{umbral}_{fecha}"
    ignicion = date(int(fecha[:4]), int(fecha[4:6]), int(fecha[6:]))
    season = temporada if temporada is not None else lfs.temporada(ignicion)
    zf.writestr(base + ".tif", cicatriz_tif(mascara, oeste, norte))
    fila = {"FireID": fid, "Season": season, "TotalArea": area_m2, "OverlapIDs": solapes}
    zf.writestr(base + ".dbf", dbf([fila, fila]))
    # Siblings the reader must ignore
    zf.writestr(f"{raiz}/RdNBR_{region}_ID{fid}_u{umbral}_{fecha}.tif", b"not read")
    zf.writestr(f"{raiz}/Severity_{region}_ID{fid}_u{umbral}_{fecha}.tif", b"not read")


@pytest.fixture
def banco_lfs(tmp_path):
    carpeta = tmp_path / "bruto" / lfs.CARPETA
    carpeta.mkdir(parents=True)
    banco.preparar(carpeta, FUENTE)
    uno = np.zeros((6, 7), dtype=np.uint8)
    uno[1:5, 2:6] = 1
    dos = np.ones((5, 5), dtype=np.uint8)
    with zipfile.ZipFile(carpeta / "FireScar_CL-VS_Valparaiso_2009-2018.zip", "w") as zf:
        raiz = "Fire_CL-VS_Valparaiso/FireScar_CL-VS_Valparaiso_2016"
        agregar_fuego(zf, raiz, "CL-VS", 101, 430, "20160105", uno, -71.98, -33.01, 3_000_000)
        agregar_fuego(zf, raiz, "CL-VS", 102, 350, "20151120", dos, -71.95, -33.02, 1_000_000,
                      solapes="101")
    with zipfile.ZipFile(carpeta / "FireScar_CL-ML_Maule_1985-2008.zip", "w",
                         compression=zipfile.ZIP_DEFLATED) as zf:
        raiz = "FireScar_CL-ML_Maule_1985-2008/Fire_CL-ML_Maule/FireScar_CL-ML_Maule_2011"
        agregar_fuego(zf, raiz, "CL-ML", 201, 500, "20101215", dos, -71.94, -33.03, 4_000_000)
        agregar_fuego(zf, raiz, "CL-ML", 202, 500, "20070301", dos, -60.0, -40.0, 9_000_000)
    for z in sorted(carpeta.glob("*.zip")):
        banco.registrar(carpeta, z.name, "prueba", lfs.PAGINA, False, "2026-11-07T00:00:00-03:00")
    return tmp_path / "bruto", carpeta


def _fuerza_bruta(mascara, oeste, norte, paso=PASO_LFS):
    salida = np.zeros((GRILLA.alto, GRILLA.ancho), dtype=bool)
    for f in range(GRILLA.alto):
        lat = GRILLA.norte - (f + 0.5) * GRILLA.paso
        r = int(np.floor((norte - lat) / paso))
        for c in range(GRILLA.ancho):
            lon = GRILLA.oeste + (c + 0.5) * GRILLA.paso
            k = int(np.floor((lon - oeste) / paso))
            if 0 <= r < mascara.shape[0] and 0 <= k < mascara.shape[1]:
                salida[f, c] = mascara[r, k] == 1
    return salida


def test_season_runs_july_to_june():
    assert lfs.temporada(date(2015, 6, 30)) == 2015
    assert lfs.temporada(date(2015, 7, 1)) == 2016
    assert lfs.temporada(date(2016, 1, 5)) == 2016


@pytest.mark.parametrize(
    ("ignicion", "esperado"),
    [
        (date(2007, 12, 31), lfs.BIT_HASTA_2018),
        (date(2008, 1, 1), lfs.BIT_HASTA_2018 | lfs.BIT_IGNICION_2008_2012),
        (date(2012, 12, 31), lfs.BIT_HASTA_2018 | lfs.BIT_IGNICION_2008_2012),
        (date(2013, 1, 1), lfs.BIT_HASTA_2018),
    ],
)
def test_ignition_window_includes_both_ends(ignicion, esperado):
    assert lfs.banderas_de(ignicion) == esperado


def test_dbf_reader_skips_deleted_records():
    filas = [{"FireID": 1, "Season": 2016}, {"FireID": 2, "Season": 2017}]
    leidas = lfs.leer_dbf(dbf(filas, borrados=(0,)))
    assert leidas == [{"FireID": "2", "Season": "2017", "TotalArea": "", "OverlapIDs": ""}]
    with pytest.raises(SourceError):
        lfs.leer_dbf(b"\x03" * 10)


@pytest.mark.parametrize(
    ("oeste", "norte"), [(-71.98, -33.01), (-72.003, -32.999), (-71.93, -33.05)]
)
def test_projection_takes_the_scar_pixel_holding_each_centre(oeste, norte):
    rng = np.random.default_rng(3)
    mascara = (rng.random((9, 11)) < 0.5).astype(np.uint8)
    f0, c0, ventana = lfs.proyectar(cicatriz_tif(mascara, oeste, norte), GRILLA)
    obtenido = np.zeros((GRILLA.alto, GRILLA.ancho), dtype=bool)
    obtenido[f0 : f0 + ventana.shape[0], c0 : c0 + ventana.shape[1]] = ventana
    np.testing.assert_array_equal(obtenido, _fuerza_bruta(mascara, oeste, norte))
    assert obtenido.any()


def test_projection_outside_the_grid_is_empty():
    _, _, ventana = lfs.proyectar(cicatriz_tif(np.ones((3, 3)), -60.0, -40.0), GRILLA)
    assert ventana.size == 0


def test_projection_refuses_other_systems():
    contenido = tif(np.ones((1, 3, 3)), ("b1",), oeste=300000, norte=6300000, paso=30,
                    crs="EPSG:32719", sin_dato=None)
    with pytest.raises(SourceError, match="EPSG:4326"):
        lfs.proyectar(contenido, GRILLA)


def test_build_flags_inventory_and_comparison_pixels(banco_lfs, tmp_path):
    bruto, _ = banco_lfs
    salida = tmp_path / "lfs"
    registro = lfs.construir(bruto, GRILLA, salida, temporadas_comparacion=(2016,),
                             area_minima_ha=250.0)
    assert registro["cicatrices"] == 4
    assert registro["cicatrices_en_grilla"] == 3
    assert registro["comparacion"]["cicatrices"] == 1  # 101 (300 ha); 102 has 100 ha
    inventario = pq.read_table(salida / "inventario.parquet").to_pylist()
    por_id = {f["id"]: f for f in inventario}
    assert [f["id"] for f in inventario] == [101, 102, 201, 202]
    assert por_id[102]["solapes"] == "101" and por_id[102]["temporada"] == 2016
    assert por_id[201]["ignicion"] == "2010-12-15" and por_id[201]["temporada"] == 2011
    assert por_id[202]["pixeles_en_grilla"] == 0
    assert por_id[101]["zip"] == "FireScar_CL-VS_Valparaiso_2009-2018.zip"
    banderas = lfs.leer_banderas(salida, GRILLA, 0, GRILLA.alto)
    uno = np.zeros((6, 7), dtype=np.uint8)
    uno[1:5, 2:6] = 1
    dos = np.ones((5, 5), dtype=np.uint8)
    a = _fuerza_bruta(uno, -71.98, -33.01)
    b = _fuerza_bruta(dos, -71.95, -33.02)
    m = _fuerza_bruta(dos, -71.94, -33.03)
    np.testing.assert_array_equal((banderas & lfs.BIT_HASTA_2018) != 0, a | b | m)
    np.testing.assert_array_equal((banderas & lfs.BIT_IGNICION_2008_2012) != 0, m)
    pixeles = pq.read_table(salida / "pixeles_comparacion.parquet").to_pylist()
    assert {p["id"] for p in pixeles} == {101}
    assert len(pixeles) == int(a.sum())
    for p in pixeles:
        assert a[p["fila"], p["columna"]]
    leido = json.loads((salida / "lfs.json").read_text(encoding="utf-8"))
    assert leido["grilla"] == asdict(GRILLA)
    for nombre, valor in leido["sha256"].items():
        assert hashlib.sha256((salida / nombre).read_bytes()).hexdigest() == valor
    with pytest.raises(SourceError, match="overwrite"):
        lfs.construir(bruto, GRILLA, salida)


def test_area_threshold_is_strict(banco_lfs, tmp_path):
    bruto, _ = banco_lfs
    lfs.construir(bruto, GRILLA, tmp_path / "a", temporadas_comparacion=(2016,),
                  area_minima_ha=100.0)
    elegidas = {
        f["id"] for f in pq.read_table(tmp_path / "a" / "inventario.parquet").to_pylist()
        if f["comparacion"]
    }
    assert elegidas == {101}  # 102 has exactly 100 ha


def test_unrecorded_zip_is_refused(banco_lfs, tmp_path):
    bruto, carpeta = banco_lfs
    with zipfile.ZipFile(carpeta / "extra.zip", "w") as zf:
        zf.writestr("x.txt", b"x")
    with pytest.raises(SourceError, match="not recorded"):
        lfs.construir(bruto, GRILLA, tmp_path / "salida")


def test_season_must_match_the_ignition_date(tmp_path):
    carpeta = tmp_path / "bruto" / lfs.CARPETA
    carpeta.mkdir(parents=True)
    banco.preparar(carpeta, FUENTE)
    with zipfile.ZipFile(carpeta / "a.zip", "w") as zf:
        agregar_fuego(zf, "r", "CL-VS", 1, 400, "20160105", np.ones((3, 3)), -71.98, -33.01,
                      1e6, temporada=2015)
    banco.registrar(carpeta, "a.zip", "p", lfs.PAGINA, False, "2026-11-07T00:00:00-03:00")
    with pytest.raises(SourceError, match="season 2015"):
        lfs.construir(tmp_path / "bruto", GRILLA, tmp_path / "salida")


def test_duplicate_scar_is_refused(tmp_path):
    carpeta = tmp_path / "bruto" / lfs.CARPETA
    carpeta.mkdir(parents=True)
    banco.preparar(carpeta, FUENTE)
    for nombre in ("a.zip", "b.zip"):
        with zipfile.ZipFile(carpeta / nombre, "w") as zf:
            agregar_fuego(zf, "r", "CL-VS", 7, 400, "20160105", np.ones((3, 3)), -71.98, -33.01,
                          1e6)
        banco.registrar(carpeta, nombre, "p", lfs.PAGINA, False, "2026-11-07T00:00:00-03:00")
    with pytest.raises(SourceError, match="twice"):
        lfs.construir(tmp_path / "bruto", GRILLA, tmp_path / "salida")


def test_flags_from_another_grid_are_refused(banco_lfs, tmp_path):
    bruto, _ = banco_lfs
    lfs.construir(bruto, GRILLA, tmp_path / "s")
    otra = exportacion.Grilla(0.0, 0.0, PASO, GRILLA.ancho, GRILLA.alto)
    with pytest.raises(SourceError, match="another grid"):
        lfs.leer_banderas(tmp_path / "s", otra, 0, 1)
