import json

import h3.api.numpy_int as h3n
import numpy as np
import pyarrow.parquet as pq
import pytest

from nadir import banco
from nadir.__main__ import main
from nadir.capas import cobertura, cobertura_c1, elevacion, fuego
from nadir.errors import GridError, SourceError
from nadir.estudio import analisis, categorias, tablas
from sinteticos import PASO, exportar
from test_estudio_analisis import _escribir
from test_estudio_tablas import ALTO, ANCHO, NORTE, OESTE, _leer, entorno  # noqa: F401


def _exportar_c1(raiz, cob1, oeste=OESTE):
    carpeta = raiz / cobertura_c1.CARPETA
    banco.preparar(carpeta, cobertura_c1.FUENTE)
    exportar(carpeta / "c1.zip", cobertura_c1.PREFIJO, cob1,
             tuple(cobertura_c1.banda(a) for a in cobertura_c1.ANIOS_EXPORTADOS), lado=8,
             oeste=oeste, norte=NORTE)
    banco.registrar(carpeta, "c1.zip", "2012-2022", "earthengine:x", False)


def _cubo_c1(semilla=11):
    rng = np.random.default_rng(semilla)
    cob1 = rng.choice([3, 4, 9, 12, 13, 21, 24, 66], size=(11, ALTO, ANCHO)).astype(np.uint8)
    cob1[:, 0, :] = 255
    cob1[3, 5, 5] = 27
    cob1[6, 7, 7] = 0
    return cob1


def _esperado_c1(g, cob1, anual, tramos):
    areas = g.area_filas_m2(0, ALTO)
    veces = (anual == 1).sum(axis=0)
    tabla = {}
    for anio in cobertura_c1.ANIOS_FUEGO:
        b = fuego.ANIOS.index(anio)
        for f in range(ALTO):
            for c in range(ANCHO):
                if anual[0, f, c] == 255 or tramos[f, c] == elevacion.SIN_TRAMO:
                    continue
                codigos = tuple(
                    int(cob1[cobertura_c1.ANIOS_EXPORTADOS.index(anio + d), f, c])
                    for d in (-1, 2, 3)
                )
                if any(x in (0, 255, 27) for x in codigos):
                    continue
                if veces[f, c] == 0:
                    grupo = 0
                elif anual[b, f, c] == 1:
                    grupo = 1 if veces[f, c] == 1 else 2
                else:
                    continue
                lat = float(g.latitudes(f, 1)[0])
                lon = float(g.longitudes(c, 1)[0])
                celda = int(h3n.cell_to_parent(h3n.latlng_to_cell(lat, lon, 8), 7))
                clave = (celda, anio, grupo, int(tramos[f, c]), *codigos)
                area, n = tabla.get(clave, (0.0, 0))
                tabla[clave] = (area + areas[f], n + 1)
    return tabla


def test_collection_1_table_matches_a_pixel_by_pixel_count(entorno, tmp_path):  # noqa: F811
    raiz, insumos, g, _, anual, tramos, _ = entorno
    cob1 = _cubo_c1()
    _exportar_c1(raiz, cob1)
    huellas = tablas.construir(raiz, insumos, tmp_path / "t1", ("k3",), coleccion=1)
    assert set(huellas) == {"k3"}
    t = pq.read_table(tmp_path / "t1" / "k3.parquet")
    assert t.schema.metadata[b"coleccion"] == b"1"
    assert set(t.column("anio").to_pylist()) <= set(cobertura_c1.ANIOS_FUEGO)
    obtenido = {k[:4] + k[5:]: v for k, v in _leer(tmp_path / "t1" / "k3.parquet", True, 3).items()}
    esperado = _esperado_c1(g, cob1, anual, tramos)
    assert obtenido.keys() == esperado.keys()
    for clave, (area, n) in esperado.items():
        assert obtenido[clave][0] == pytest.approx(area, rel=1e-9)
        assert obtenido[clave][1] == n


def test_collection_1_builds_k3_only_and_checks_the_collection(entorno, tmp_path):  # noqa: F811
    raiz, insumos, *_ = entorno
    _exportar_c1(raiz, _cubo_c1())
    with pytest.raises(SourceError, match="only k3"):
        tablas.construir(raiz, insumos, tmp_path / "a", ("k3", "balance"), coleccion=1)
    with pytest.raises(SourceError, match="collections are"):
        tablas.construir(raiz, insumos, tmp_path / "b", ("k3",), coleccion=3)


def test_unknown_code_stops_the_reading(entorno, tmp_path):  # noqa: F811
    raiz, insumos, *_ = entorno
    cob1 = _cubo_c1()
    cob1[4, 8, 8] = 2  # a network code with no level-1 class in the table
    _exportar_c1(raiz, cob1)
    with pytest.raises(SourceError, match=r"outside the level-1 table of collection 1: \[2\]"):
        tablas.construir(raiz, insumos, tmp_path / "t", ("k3",), coleccion=1)
    with cobertura_c1.abrir(raiz) as m, pytest.raises(SourceError, match=r"\[2\]"):
        cobertura_c1.describir(m)


def test_another_grid_is_refused(entorno):  # noqa: F811
    raiz, *_ = entorno
    _exportar_c1(raiz, _cubo_c1(), oeste=OESTE + PASO)
    with pytest.raises(GridError, match="not resampled"):
        cobertura_c1.abrir(raiz)


def test_level_1_classes_agree_with_collection_2():
    nivel = {
        "formacion_boscosa": 1,
        "formacion_natural_no_boscosa": 2,
        "agropecuaria_y_silvicultura": 3,
        "area_sin_vegetacion": 4,
        "cuerpo_de_agua": 5,
    }
    for codigo, clase in cobertura.NIVEL_1.items():
        assert cobertura_c1.NIVEL_1[codigo] == clase
    for codigo, clase in cobertura_c1.NIVEL_1.items():
        if clase in nivel:
            cat = categorias.CATEGORIA_C1[codigo]
            assert categorias.CATEGORIA_A_NIVEL1[cat] == nivel[clase], codigo
    assert (categorias.CATEGORIA_C1[list(cobertura.NIVEL_1)]
            == categorias.CATEGORIA[list(cobertura.NIVEL_1)]).all()


def test_load_uses_the_collection_of_the_table(tmp_path):
    carpeta = _escribir(tmp_path / "t")
    k3 = pq.read_table(carpeta / "k3.parquet")
    c0 = np.where(np.arange(k3.num_rows) % 7 == 0, 21, k3.column("c0").to_numpy())
    k3 = k3.set_column(k3.schema.get_field_index("c0"), "c0", [c0.astype(np.uint8)])
    pq.write_table(k3, carpeta / "k3.parquet")
    sin = categorias.cargar(carpeta / "k3.parquet")
    assert 0 in set(sin.c0)  # collection 2 has no code 21
    pq.write_table(k3.replace_schema_metadata({"coleccion": "1"}), carpeta / "k3.parquet")
    con = categorias.cargar(carpeta / "k3.parquet")
    assert 0 not in set(con.c0) and 6 in set(con.c0)


def test_sensitivity_7(tmp_path, capsys):
    c2 = _escribir(tmp_path / "c2", semilla=1)
    c1 = _escribir(tmp_path / "c1", semilla=2)
    with pytest.raises(SourceError, match="is not a coleccion 1 table"):
        analisis.sensibilidad_coleccion(c2, c1, replicas=5)
    k3 = pq.read_table(c1 / "k3.parquet")
    pq.write_table(k3.replace_schema_metadata({"coleccion": "1",
                                               "excluido_codigo_no_valido_m2": "250000.0"}),
                   c1 / "k3.parquet")
    with pytest.raises(SourceError, match="is not a coleccion 2 table"):
        analisis.sensibilidad_coleccion(c1, c1, replicas=5)
    r = analisis.sensibilidad_coleccion(c2, c1, replicas=5)
    assert r["anios_fuego"] == [2013, 2019]
    h1_1, h1_2 = r["7_H1_coleccion_1"], r["7_H1_coleccion_2"]
    assert h1_1["excluido_codigo_no_valido_ha_sumado_por_anio"] == 25.0
    assert r["diferencia_c1_menos_c2_pp_descriptiva"] == round(
        h1_1["efecto_pp"] - h1_2["efecto_pp"], 2
    )
    otra = analisis.sensibilidad_coleccion(c2, c1, replicas=5)
    assert json.dumps(otra) == json.dumps(r)
    salida = tmp_path / "s7.json"
    assert main(["estudio", "sensibilidad7", "--tablas", str(c2), "--tablas-c1", str(c1),
                 "--salida", str(salida), "--replicas", "5"]) == 0
    assert "7_H1_coleccion_1: effect" in capsys.readouterr().out
    assert json.loads(salida.read_text(encoding="utf-8"))["replicas"] == 5


def test_command_line_prepares_and_checks_the_export(entorno, capsys):  # noqa: F811
    raiz, *_ = entorno
    assert main(["cobertura-c1", "preparar", "--bruto", str(raiz)]) == 0
    _exportar_c1(raiz, _cubo_c1())
    assert main(["cobertura-c1", "comprobar", "--bruto", str(raiz), "--anios", "2012"]) == 0
    texto = capsys.readouterr().out
    assert "same grid as collection 2" in texto
    assert "agropecuaria_y_silvicultura" in texto and " 21 " in texto
