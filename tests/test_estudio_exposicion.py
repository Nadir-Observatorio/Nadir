import json

import h3.api.numpy_int as h3n
import numpy as np
import pandas as pd
import pytest
from shapely import to_wkb
from shapely.geometry import Polygon, box

from nadir import banco, celdas
from nadir.capas import cobertura, interfaz
from nadir.errors import SourceError
from nadir.estudio import exposicion
from nadir.estudio.categorias import CATEGORIA, CATEGORIA_A_NIVEL1
from sinteticos import exportar

PASO = 0.01
OESTE, NORTE = -72.0, -35.5
ALTO, ANCHO = 30, 40
BANDAS = tuple(cobertura.banda(a) for a in cobertura.ANIOS_ESTUDIO)


def _banda_fija(valor):
    return lambda lat, lon: np.full(np.shape(lat), valor, dtype=np.uint8)


def test_population_is_shared_by_area_and_side():
    centro = int(h3n.latlng_to_cell(-35.6, -71.8, 6))
    vecina = next(int(c) for c in h3n.grid_ring(centro, 1))
    hex_c = Polygon([(lon, lat) for lat, lon in h3n.cell_to_boundary(centro)])
    hex_v = Polygon([(lon, lat) for lat, lon in h3n.cell_to_boundary(vecina)])
    dentro = hex_c.buffer(-0.005)  # wholly in the centre cell
    cruza = hex_c.union(hex_v).envelope  # spills over both cells and beyond
    censo = pd.DataFrame({"n_per": [100.0, 1000.0], "n_vp": [40.0, 400.0],
                          "SHAPE": [to_wkb(dentro), to_wkb(cruza)]})
    p = exposicion.poblacion(censo, [centro, vecina], _banda_fija(0))
    assert set(p.lado) == {"dentro"}
    total = p.n_per.sum()
    share = (hex_c.union(hex_v).intersection(cruza)).area / cruza.area
    assert total == pytest.approx(100.0 + 1000.0 * share, rel=1e-9)
    assert p.n_vp.sum() == pytest.approx(40.0 + 400.0 * share, rel=1e-9)
    solo = exposicion.poblacion(censo, [centro], _banda_fija(1))
    assert set(solo.lado) == {"fuera"}
    assert solo.n_per.sum() == pytest.approx(100.0 + 1000.0 * hex_c.intersection(cruza).area
                                              / cruza.area, rel=1e-9)
    nada = exposicion.poblacion(censo, [int(h3n.latlng_to_cell(-20.0, -70.0, 6))], _banda_fija(0))
    assert nada.empty


def test_establishments_count_by_cell_and_side():
    a = int(h3n.latlng_to_cell(-35.6, -71.9, 6))
    lat_a, lon_a = h3n.cell_to_latlng(a)
    puntos = {
        "salud": (np.array([lat_a, lat_a, -20.0]), np.array([lon_a, lon_a, -70.0])),
        "bomberos": (np.array([lat_a]), np.array([lon_a])),
    }
    e = exposicion.establecimientos(puntos, [a], _banda_fija(0))
    assert dict(zip(e.tipo, e.n, strict=True)) == {"bomberos": 1, "salud": 2}
    e2 = exposicion.establecimientos(puntos, [a], _banda_fija(255))
    assert set(e2.lado) == {"sin_banda"}


def test_suppression_blanks_small_counts_with_their_dwellings():
    t = pd.DataFrame({"n_per_dentro": [0.0, 49.9, 50.0], "n_vp_dentro": [0.0, 20.0, 21.0],
                      "n_per_fuera": [10.0, 300.0, 0.0], "n_vp_fuera": [3.0, 90.0, 0.0]})
    s, n = exposicion.suprimir(t)
    assert n == 2
    assert s.n_per_dentro.tolist()[0] == 0.0 and np.isnan(s.n_per_dentro[1])
    assert np.isnan(s.n_vp_dentro[1]) and s.n_vp_dentro[2] == 21.0
    assert np.isnan(s.n_per_fuera[0]) and np.isnan(s.n_vp_fuera[0])


@pytest.fixture
def banco_cobertura(tmp_path):
    rng = np.random.default_rng(9)
    raiz = tmp_path / "bruto"
    carpeta = raiz / cobertura.CARPETA
    banco.preparar(carpeta, cobertura.FUENTE)
    cob = rng.choice([3, 4, 9, 12, 15, 24, 33], size=(len(BANDAS), ALTO, ANCHO)).astype(np.uint8)
    cob[:, 0, :] = cobertura.FUERA
    exportar(carpeta / "c.zip", cobertura.PREFIJO, cob, BANDAS, lado=16, paso=PASO, oeste=OESTE,
             norte=NORTE)
    banco.registrar(carpeta, "c.zip", "2012-2024", "earthengine:x", False)
    with cobertura.abrir(raiz) as m:
        celdas.construir(m, cobertura.banda(2024), 8, tmp_path / "celdas")
        interfaz.construir(m, tmp_path / "interfaz", (exposicion.ANIO_BANDA,))
        g = m.grilla
    banda = rng.integers(0, 2, (ALTO, ANCHO)).astype(np.uint8)
    banda[0, :] = interfaz.FUERA
    (tmp_path / "interfaz" / f"banda_{exposicion.ANIO_BANDA}.u8").write_bytes(banda.tobytes())
    return raiz, g, cob[BANDAS.index(cobertura.banda(exposicion.ANIO_COBERTURA))], banda


def test_land_cover_areas_by_cell_and_side(banco_cobertura, tmp_path):
    raiz, g, clase, banda = banco_cobertura
    lat = g.latitudes(0, g.alto)
    lon = g.longitudes(0, g.ancho)
    r6 = np.array([[h3n.cell_to_parent(h3n.latlng_to_cell(a, b, 8), 6) for b in lon] for a in lat])
    sel = list(np.unique(r6[1:]))[::2]
    with cobertura.abrir(raiz) as m:
        mapa = celdas.abrir(tmp_path / "celdas", m.grilla)
        t = exposicion.superficies(m, mapa, tmp_path / "interfaz", sel, filas_franja=7)
    areas = np.broadcast_to(g.area_filas_m2(0, g.alto)[:, None], clase.shape)
    nivel = CATEGORIA_A_NIVEL1[CATEGORIA[clase]]
    for celda in sel:
        for v, lado in ((0, "dentro"), (1, "fuera")):
            m_ = (r6 == celda) & (banda == v) & (clase != cobertura.FUERA)
            fila = t[(t.celda == celda) & (t.lado == lado)]
            if not m_.any():
                assert fila.empty
                continue
            assert fila.total_ha.item() == pytest.approx(areas[m_].sum() / 1e4)
            for col, n1 in exposicion.NIVEL1.items():
                assert fila[f"{col}_ha"].item() == pytest.approx(
                    areas[m_ & (nivel == n1)].sum() / 1e4)
    assert set(t.celda) <= set(sel)


def test_census_reader_checks_the_bank_and_the_values(tmp_path):
    import io
    import zipfile

    import pyarrow as pa
    import pyarrow.parquet as pq

    def zip_censo(ruta, n_per):
        with zipfile.ZipFile(ruta, "w") as z:
            for capa in exposicion.CAPAS_CENSO:
                tabla = pa.table({"n_per": n_per, "n_vp": [1.0] * len(n_per),
                                  "SHAPE": [to_wkb(box(0, 0, 1, 1))] * len(n_per)})
                b = io.BytesIO()
                pq.write_table(tabla, b)
                z.writestr(f"Cartografia_censo2024_R08_{capa}.parquet", b.getvalue())

    raiz = tmp_path / "bruto"
    for i, carpeta in enumerate(exposicion.CARPETAS_CENSO):
        banco.preparar(raiz / carpeta, banco.Fuente("INE", "Censo", "https://x", "CC-BY-4.0", "x"))
        zip_censo(raiz / carpeta / f"c{i}.zip", [5.0, 6.0])
        banco.registrar(raiz / carpeta, f"c{i}.zip", "2024", "https://x", False)
    censo = exposicion.leer_censo(raiz)
    assert len(censo) == 2 * 2 * len(exposicion.CARPETAS_CENSO)
    assert set(censo.capa) == set(exposicion.CAPAS_CENSO)
    otra = tmp_path / "otra"
    for i, carpeta in enumerate(exposicion.CARPETAS_CENSO):
        banco.preparar(otra / carpeta, banco.Fuente("INE", "Censo", "https://x", "CC-BY-4.0", "x"))
        zip_censo(otra / carpeta / f"c{i}.zip", [5.0, -1.0])
        banco.registrar(otra / carpeta, f"c{i}.zip", "2024", "https://x", False)
    with pytest.raises(SourceError, match="negative"):
        exposicion.leer_censo(otra)


def test_tables_join_every_dimension_and_write_once(tmp_path):
    a = int(h3n.latlng_to_cell(-35.6, -71.9, 6))
    b = int(h3n.latlng_to_cell(-36.6, -72.4, 6))
    sin_retorno = pd.DataFrame({"celda": [a, b], "efecto_pp": [12.0, 3.0],
                                "bosque_comparable_ha": [150.0, 400.0]})
    pob = pd.DataFrame({"celda": [a, a, b], "lado": ["dentro", "fuera", "fuera"],
                        "n_per": [30.0, 500.0, 80.0], "n_vp": [10.0, 200.0, 30.0]})
    est = pd.DataFrame({"celda": [a], "lado": ["dentro"], "tipo": ["salud"], "n": [2]})
    sup = pd.DataFrame({"celda": [a, b], "lado": ["fuera", "fuera"],
                        "ecosistema_bosque_ha": [10.0, 20.0],
                        "ecosistema_no_boscosa_ha": [1.0, 2.0],
                        "productivo_agro_silvicola_ha": [5.0, 6.0], "total_ha": [16.0, 28.0]})
    celda_t, region_t, conteo = exposicion.tablas_h5(sin_retorno, pob, est, sup,
                                                     {a: "Maule", b: "Maule"})
    assert conteo == {"suprimidas_celda": 1, "suprimidas_region": 1}
    fila = celda_t[celda_t.h3 == h3n.int_to_str(a)].iloc[0]
    assert np.isnan(fila.n_per_dentro) and fila.n_per_fuera == 500.0
    assert fila.salud_dentro == 2 and fila.ecosistema_bosque_ha_fuera == 10.0
    assert fila.bomberos_fuera == 0 and "sin_banda" not in "".join(celda_t.columns)
    r = region_t.iloc[0]
    assert r.region == "Maule" and r.celdas == 2
    assert np.isnan(r.n_per_dentro)  # 30 persons in the region inside: suppressed too
    assert r.n_per_fuera == 580.0 and r.productivo_agro_silvicola_ha_fuera == 11.0
    meta = {"nota": exposicion.NOTA}
    rutas = exposicion.escribir(celda_t, region_t, meta, tmp_path / "h5")
    assert json.loads(rutas[2].read_text(encoding="utf-8"))["nota"].startswith("aggregates")
    with pytest.raises(SourceError, match="overwrite"):
        exposicion.escribir(celda_t, region_t, meta, tmp_path / "h5")


def test_band_reader_takes_the_pixel_under_each_point(banco_cobertura, tmp_path):
    _, g, _, banda = banco_cobertura
    leer = exposicion.lector_banda(tmp_path / "interfaz", g, exposicion.ANIO_BANDA)
    filas = np.array([0, 3, 3, 29])
    cols = np.array([5, 0, 39, 17])
    lat = g.latitudes(0, g.alto)[filas]
    lon = g.longitudes(0, g.ancho)[cols]
    assert (leer(lat, lon) == banda[filas, cols]).all()
    afuera = leer(np.array([NORTE + 1.0]), np.array([OESTE + 0.1]))
    assert afuera.tolist() == [interfaz.FUERA]


def test_command_line_builds_the_band_of_the_exposure_year(banco_cobertura, tmp_path, capsys):
    from nadir.__main__ import main

    raiz = banco_cobertura[0]
    assert main(["interfaz", "construir", "--bruto", str(raiz), "--salida",
                 str(tmp_path / "i2"), "--anios", str(exposicion.ANIO_BANDA)]) == 0
    assert (tmp_path / "i2" / f"banda_{exposicion.ANIO_BANDA}.u8").exists()
    assert main(["interfaz", "construir", "--bruto", str(raiz), "--salida",
                 str(tmp_path / "i3"), "--anios", "2022"]) == 1
    assert "years outside" in capsys.readouterr().err


def test_provenance_entries_from_other_projects_are_read_leniently(tmp_path):
    import hashlib

    carpeta = tmp_path / "f"
    carpeta.mkdir()
    (carpeta / "a.zip").write_bytes(b"datos")
    huella = hashlib.sha256(b"datos").hexdigest()
    (carpeta / "procedencia.toml").write_text(
        '[fuente]\norganismo = "INE"\nnota = "otra forma"\n\n'
        '[[archivo]]\nnombre = "á.zip"\nbytes = 5\nsha256 = "x"\n\n'
        f'[[archivo]]\nnombre = "a.zip"\nbytes = 5\nsha256 = "{huella}"\nnota = "extra"\n',
        encoding="utf-8")
    assert exposicion.registrados(carpeta) == ["a.zip"]
    (carpeta / "a.zip").write_bytes(b"otros")
    with pytest.raises(SourceError, match="does not match"):
        exposicion.registrados(carpeta)
    with pytest.raises(SourceError, match="no provenance"):
        exposicion.registrados(tmp_path)
