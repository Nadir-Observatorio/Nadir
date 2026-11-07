import json

import h3.api.numpy_int as h3n
import numpy as np
import pyarrow.parquet as pq
import pytest

from nadir import celdas, exportacion, transiciones
from nadir.capas import cobertura, fuego
from nadir.errors import LayerError, TableError
from sinteticos import exportar

PASO = 0.002
ALTO, ANCHO = 24, 30
COB_BANDAS = tuple(cobertura.banda(a) for a in cobertura.ANIOS_ESTUDIO)
FUE_BANDAS = tuple(fuego.banda_anual(a) for a in fuego.ANIOS)


@pytest.fixture
def datos(tmp_path):
    rng = np.random.default_rng(11)
    cob = rng.choice([3, 9, 12, 24, 66], size=(13, ALTO, ANCHO)).astype(np.uint8)
    cob[:, :2, :] = 255
    cob[4, 5, 5] = 0
    cob[6, 6, 6] = 27
    anual = (rng.random((13, ALTO, ANCHO)) < 0.08).astype(np.uint8)
    anual[:, :2, :] = 255
    anual[:, 2, :4] = 255  # inside land cover, outside the fire export
    anual[:, 5, 5] = 0  # never burned: the 0 and 27 codes above land in the control group
    anual[:, 6, 6] = 0
    rc = exportar(tmp_path / "c.zip", "c", cob, COB_BANDAS, lado=16, paso=PASO)
    rf = exportar(tmp_path / "f.zip", "f", anual, FUE_BANDAS, lado=16, paso=PASO)
    mc = exportacion.abrir([rc], "c")
    mf = exportacion.abrir([rf], "f")
    mapa = celdas.construir(mc, COB_BANDAS[-1], 8, tmp_path / "mapa")
    yield mapa, mc, mf, cob, anual
    mc.cerrar()
    mf.cerrar()


def _fuerza_bruta(mapa, cob, anual, anios, desfases, resolucion):
    g = mapa.grilla
    areas = g.area_filas_m2(0, ALTO)
    veces = (anual == 1).sum(axis=0)
    esperado = {}
    for anio in anios:
        b = fuego.ANIOS.index(anio)
        for f in range(ALTO):
            for c in range(ANCHO):
                if cob[-1, f, c] == 255 or anual[0, f, c] == 255:
                    continue
                codigos = [
                    int(cob[cobertura.ANIOS_ESTUDIO.index(anio + d), f, c]) for d in desfases
                ]
                if any(x in (0, 255, 27) for x in codigos):
                    continue
                if veces[f, c] == 0:
                    grupo = "control"
                elif anual[b, f, c] == 1:
                    grupo = "quemado" if veces[f, c] == 1 else "requemado"
                else:
                    continue
                lat = float(g.latitudes(f, 1)[0])
                lon = float(g.longitudes(c, 1)[0])
                celda = h3n.cell_to_parent(h3n.latlng_to_cell(lat, lon, 8), resolucion)
                clave = (celda, anio, grupo, tuple(codigos))
                esperado[clave] = esperado.get(clave, 0.0) + areas[f]
    return esperado


def _como_dict(tabla):
    return {
        (
            int(tabla.celdas[i]),
            int(tabla.anio[i]),
            transiciones.GRUPOS[int(tabla.grupo[i])],
            tuple(int(x) for x in tabla.clases[i]),
        ): float(tabla.area_m2[i])
        for i in range(len(tabla))
    }


@pytest.mark.parametrize(("resolucion", "alto"), [(8, 5), (7, 512), (6, 1)])
def test_table_matches_a_pixel_by_pixel_count(datos, resolucion, alto):
    mapa, mc, mf, cob, anual = datos
    anios, desfases = (2014, 2016), (-1, 2, 3)
    tabla = transiciones.tabular(mapa, mc, mf, anios, desfases, resolucion, alto)
    obtenido = _como_dict(tabla)
    esperado = _fuerza_bruta(mapa, cob, anual, anios, desfases, resolucion)
    assert obtenido.keys() == esperado.keys()
    for clave, valor in esperado.items():
        assert obtenido[clave] == pytest.approx(valor, rel=1e-9)
    assert {"control", "quemado", "requemado"} <= {k[2] for k in obtenido}


def test_offsets_keep_the_order_given(datos):
    mapa, mc, mf, cob, anual = datos
    tabla = transiciones.tabular(mapa, mc, mf, (2016,), (2, -1), 8)
    esperado = _fuerza_bruta(mapa, cob, anual, (2016,), (2, -1), 8)
    assert _como_dict(tabla).keys() == esperado.keys()


def test_coarser_tables_add_up(datos):
    mapa, mc, mf, *_ = datos
    fina = transiciones.tabular(mapa, mc, mf, (2015,), (-1, 1, 2), 8)
    gruesa = transiciones.tabular(mapa, mc, mf, (2015,), (-1, 1, 2), 5)
    assert float(fina.area_m2.sum()) == pytest.approx(float(gruesa.area_m2.sum()), rel=1e-12)
    assert set(gruesa.celdas.tolist()) == {h3n.cell_to_parent(int(c), 5) for c in fina.celdas}


def test_offsets_and_years_are_checked(datos):
    mapa, mc, mf, *_ = datos
    with pytest.raises(LayerError, match="needs land cover 2025"):
        transiciones.tabular(mapa, mc, mf, (2022,), (-1, 2, 3))
    with pytest.raises(LayerError, match="needs land cover 2011"):
        transiciones.tabular(mapa, mc, mf, (2012 + 1,), (-2,))
    with pytest.raises(LayerError, match="distinct"):
        transiciones.tabular(mapa, mc, mf, (2015,), (1, 1))
    with pytest.raises(LayerError, match="distinct"):
        transiciones.tabular(mapa, mc, mf, (), (-1,))
    with pytest.raises(LayerError, match="strip height"):
        transiciones.tabular(mapa, mc, mf, (2015,), (-1,), alto=0)


def test_only_the_previous_class_for_recent_fires(datos):
    mapa, mc, mf, cob, anual = datos
    tabla = transiciones.tabular(mapa, mc, mf, (2025,), (-1,), 8)
    esperado = _fuerza_bruta(mapa, cob, anual, (2025,), (-1,), 8)
    assert _como_dict(tabla).keys() == esperado.keys()


def test_grids_must_match(datos, tmp_path):
    mapa, mc, _, _, anual = datos
    otra = exportar(tmp_path / "o.zip", "o", anual, FUE_BANDAS, lado=16, paso=PASO, oeste=-71.0)
    with exportacion.abrir([otra], "o") as mo, pytest.raises(LayerError, match="one grid"):
        transiciones.tabular(mapa, mc, mo, (2015,), (-1,))


def test_written_table_has_codes_level_1_and_its_hash(datos, tmp_path):
    mapa, mc, mf, *_ = datos
    tabla = transiciones.tabular(mapa, mc, mf, (2014,), (-1, 2, 3), 7)
    ruta = tmp_path / "t.parquet"
    huella = transiciones.escribir(ruta, tabla, {"cobertura": "abc"})
    assert huella == transiciones.huella(tabla)
    leida = pq.read_table(ruta)
    assert leida.column_names == [
        "celda",
        "anio_fuego",
        "grupo",
        "clase_tmenos1",
        "nivel1_tmenos1",
        "clase_tmas2",
        "nivel1_tmas2",
        "clase_tmas3",
        "nivel1_tmas3",
        "area_m2",
    ]
    meta = leida.schema.metadata
    assert meta[b"huella"].decode() == huella
    assert json.loads(meta[b"desfases"]) == [-1, 2, 3]
    assert json.loads(meta[b"procedencia"]) == {"cobertura": "abc"}
    fila = leida.slice(0, 1).to_pylist()[0]
    assert fila["nivel1_tmenos1"] == cobertura.NIVEL_1[fila["clase_tmenos1"]]
    assert sum(leida.column("area_m2").to_pylist()) == pytest.approx(float(tabla.area_m2.sum()))
    with pytest.raises(TableError, match="refusing to overwrite"):
        transiciones.escribir(ruta, tabla, {})


def test_rows_do_not_depend_on_strip_height(datos):
    mapa, mc, mf, *_ = datos
    a = transiciones.tabular(mapa, mc, mf, (2016,), (-1, 1, 2), 7, alto=3)
    b = transiciones.tabular(mapa, mc, mf, (2016,), (-1, 1, 2), 7, alto=512)
    assert [tuple(x) for x in a.clases] == [tuple(x) for x in b.clases]
    np.testing.assert_allclose(a.area_m2, b.area_m2, rtol=1e-9)


def _pendiente_falsa(carpeta, grilla, semilla=5):
    """Slope files as `construir_pendiente` writes them, with random bands and some gaps."""
    from dataclasses import asdict

    rng = np.random.default_rng(semilla)
    tramos = rng.integers(0, 3, size=(grilla.alto, grilla.ancho)).astype(np.uint8)
    tramos[7, :5] = 255
    carpeta.mkdir()
    (carpeta / "tramos_pendiente.u8").write_bytes(tramos.tobytes())
    (carpeta / "pendiente.json").write_text(json.dumps({"grilla": asdict(grilla)}), "utf-8")
    return tramos


def test_slope_bands_split_the_rows(datos, tmp_path):
    mapa, mc, mf, cob, anual = datos
    tramos = _pendiente_falsa(tmp_path / "pend", mapa.grilla)
    anios, desfases = (2014,), (-1, 3)
    con = transiciones.tabular(mapa, mc, mf, anios, desfases, 8, 5, pendiente=tmp_path / "pend")
    sin = transiciones.tabular(mapa, mc, mf, anios, desfases, 8)
    assert con.con_pendiente and not sin.con_pendiente
    assert set(sin.tramo.tolist()) == {3}
    assert set(con.tramo.tolist()) <= {0, 1, 2}
    g = mapa.grilla
    areas = g.area_filas_m2(0, ALTO)
    veces = (anual == 1).sum(axis=0)
    esperado: dict = {}
    excluido = 0.0
    for f in range(ALTO):
        for c in range(ANCHO):
            if cob[-1, f, c] == 255 or anual[0, f, c] == 255:
                continue
            codigos = [int(cob[cobertura.ANIOS_ESTUDIO.index(2014 + d), f, c]) for d in desfases]
            if any(x in (0, 255, 27) for x in codigos):
                continue
            b = fuego.ANIOS.index(2014)
            if veces[f, c] == 0:
                grupo = 0
            elif anual[b, f, c] == 1:
                grupo = 1 if veces[f, c] == 1 else 2
            else:
                continue
            if tramos[f, c] == 255:
                excluido += areas[f]
                continue
            lat, lon = float(g.latitudes(f, 1)[0]), float(g.longitudes(c, 1)[0])
            clave = (h3n.latlng_to_cell(lat, lon, 8), grupo, int(tramos[f, c]), tuple(codigos))
            esperado[clave] = esperado.get(clave, 0.0) + areas[f]
    obtenido = {
        (
            int(con.celdas[i]),
            int(con.grupo[i]),
            int(con.tramo[i]),
            tuple(int(x) for x in con.clases[i]),
        ): float(con.area_m2[i])
        for i in range(len(con))
    }
    assert obtenido.keys() == esperado.keys()
    for clave, valor in esperado.items():
        assert obtenido[clave] == pytest.approx(valor, rel=1e-9)
    assert con.excluido_sin_pendiente_m2 == pytest.approx(excluido, rel=1e-9)
    assert excluido > 0
    total_con = float(con.area_m2.sum()) + con.excluido_sin_pendiente_m2
    assert total_con == pytest.approx(float(sin.area_m2.sum()), rel=1e-9)


def test_slope_column_is_written_only_with_slope(datos, tmp_path):
    mapa, mc, mf, *_ = datos
    _pendiente_falsa(tmp_path / "pend", mapa.grilla)
    con = transiciones.tabular(mapa, mc, mf, (2015,), (-1,), 7, pendiente=tmp_path / "pend")
    sin = transiciones.tabular(mapa, mc, mf, (2015,), (-1,), 7)
    transiciones.escribir(tmp_path / "con.parquet", con, {})
    transiciones.escribir(tmp_path / "sin.parquet", sin, {})
    leida = pq.read_table(tmp_path / "con.parquet")
    assert leida.column_names[3] == "tramo_pendiente"
    assert set(leida.column("tramo_pendiente").to_pylist()) <= set(transiciones.TRAMOS)
    assert json.loads(leida.schema.metadata[b"con_pendiente"]) is True
    assert "tramo_pendiente" not in pq.read_table(tmp_path / "sin.parquet").column_names
    assert transiciones.huella(con) != transiciones.huella(sin)


def test_slope_from_another_grid_is_refused(datos, tmp_path):
    mapa, mc, mf, *_ = datos
    otra = exportacion.Grilla(0.0, 0.0, PASO, ANCHO, ALTO)
    _pendiente_falsa(tmp_path / "pend", otra)
    with pytest.raises(Exception, match="another grid"):
        transiciones.tabular(mapa, mc, mf, (2015,), (-1,), 8, pendiente=tmp_path / "pend")
