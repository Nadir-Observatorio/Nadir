import json
from dataclasses import asdict

import h3.api.numpy_int as h3n
import numpy as np
import pyarrow.parquet as pq
import pytest

from nadir import banco, celdas
from nadir.__main__ import main
from nadir.capas import cobertura, elevacion, fuego, interfaz
from nadir.errors import SourceError
from nadir.estudio import tablas
from sinteticos import exportar

OESTE, NORTE = -71.5, -33.5
ALTO, ANCHO = 14, 16


@pytest.fixture
def entorno(tmp_path):
    rng = np.random.default_rng(5)
    raiz = tmp_path / "bruto"
    cob_dir = raiz / cobertura.CARPETA
    banco.preparar(cob_dir, cobertura.FUENTE)
    cob = rng.choice([3, 9, 12, 15, 21, 24, 66], size=(13, ALTO, ANCHO)).astype(np.uint8)
    cob[:, 0, :] = 255
    cob[5, 4, 4] = 27
    cob[7, 6, 6] = 0
    cob[1, 9, 9] = 3
    cob[2:6, 9, 9] = 24  # code 24 in 2014-2017: the filter window of fire year 2014
    cob[7:9, 9, 9] = 3  # valid again in 2019 and 2020 (t + 4 and t + 5 for 2015)
    exportar(cob_dir / "c.zip", cobertura.PREFIJO, cob,
             tuple(cobertura.banda(a) for a in cobertura.ANIOS_ESTUDIO), lado=8,
             oeste=OESTE, norte=NORTE)
    banco.registrar(cob_dir, "c.zip", "2012-2024", "earthengine:x", False)
    fue_dir = raiz / fuego.CARPETA
    banco.preparar(fue_dir, fuego.FUENTE)
    anual = (rng.random((13, ALTO, ANCHO)) < 0.15).astype(np.uint8)
    anual[:, 0, :] = 255
    anual[:, 1, :3] = 255
    anual[:, 9, 9] = 0  # never burned: the marked pixel enters the control
    exportar(fue_dir / "f.zip", fuego.PREFIJO_ANUAL, anual,
             tuple(fuego.banda_anual(a) for a in fuego.ANIOS), lado=8, oeste=OESTE, norte=NORTE)
    exportar(fue_dir / "f.zip", fuego.PREFIJO_MENSUAL, anual * 2,
             tuple(fuego.banda_mensual(a) for a in fuego.ANIOS), lado=8, oeste=OESTE,
             norte=NORTE)
    banco.registrar(fue_dir, "f.zip", "2013-2025", "earthengine:x", False)
    with cobertura.abrir(raiz) as mosaico:
        g = mosaico.grilla
        celdas.construir(mosaico, cobertura.banda(2024), 8, tmp_path / "celdas")
        interfaz.construir(mosaico, tmp_path / "interfaz")
    # The synthetic area is smaller than 2 km: replace the layers by random values so both
    # bands and many distances occur (the reader only checks the grid).
    for anio in interfaz.ANIOS:
        (tmp_path / "interfaz" / f"banda_{anio}.u8").write_bytes(
            rng.integers(0, 2, (ALTO, ANCHO)).astype(np.uint8).tobytes())
        (tmp_path / "interfaz" / f"dist40_{anio}.u8").write_bytes(
            rng.integers(0, 251, (ALTO, ANCHO)).astype(np.uint8).tobytes())
    tramos = rng.integers(0, 3, (ALTO, ANCHO)).astype(np.uint8)
    tramos[0, :] = elevacion.SIN_TRAMO
    tramos[3, 3] = elevacion.SIN_TRAMO
    altitud = rng.integers(0, 2500, (ALTO, ANCHO)).astype("<i2")
    pend = tmp_path / "pendiente"
    pend.mkdir()
    (pend / "tramos_pendiente.u8").write_bytes(tramos.tobytes())
    (pend / "altitud_m.i16").write_bytes(altitud.tobytes())
    (pend / "pendiente.json").write_text(json.dumps({"grilla": asdict(g)}), encoding="utf-8")
    insumos = tablas.Insumos(tmp_path / "celdas", pend, tmp_path / "interfaz")
    return raiz, insumos, g, cob, anual, tramos, altitud


def _esperado(g, cob, anual, tramos, altitud, insumos, desfases, anios, con_banda):
    areas = g.area_filas_m2(0, ALTO)
    veces = (anual == 1).sum(axis=0)
    tabla, balance = {}, {}
    for anio in anios:
        b = fuego.ANIOS.index(anio)
        banda = (
            interfaz.leer_banda(insumos.interfaz, g, anio - 1, 0, ALTO)
            if anio - 1 in interfaz.ANIOS else np.zeros((ALTO, ANCHO), dtype=np.uint8)
        )
        dist = (
            interfaz.leer_dist40(insumos.interfaz, g, anio - 1, 0, ALTO)
            if anio - 1 in interfaz.ANIOS else None
        )
        for f in range(ALTO):
            for c in range(ANCHO):
                if anual[0, f, c] == 255 or tramos[f, c] == elevacion.SIN_TRAMO:
                    continue
                codigos = tuple(
                    int(cob[cobertura.ANIOS_ESTUDIO.index(anio + d), f, c]) for d in desfases
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
                extra = (int(banda[f, c]),) if con_banda else ()
                clave = (celda, anio, grupo, int(tramos[f, c]), *extra, *codigos)
                area, n = tabla.get(clave, (0.0, 0))
                tabla[clave] = (area + areas[f], n + 1)
                if dist is not None:
                    n1 = int(tablas.NIVEL1[codigos[0]])
                    cb = (celda, anio, grupo, int(tramos[f, c]), n1)
                    h = float(altitud[f, c])
                    d = float(dist[f, c]) * 40.0
                    w = areas[f]
                    v = balance.get(cb, np.zeros(5))
                    balance[cb] = v + np.array([w, w * h, w * h * h, w * d, w * d * d])
    return tabla, balance


def _leer(ruta, con_banda, n):
    t = pq.read_table(ruta).to_pydict()
    salida = {}
    for i in range(len(t["celda"])):
        extra = (t["banda"][i],) if con_banda else ()
        clave = (t["celda"][i], t["anio"][i], t["grupo"][i], t["tramo"][i], *extra,
                 *(t[f"c{j}"][i] for j in range(n)))
        salida[clave] = (t["area_m2"][i], t["n_px"][i])
    return salida


def test_tables_match_a_pixel_by_pixel_count(entorno, tmp_path, monkeypatch):
    raiz, insumos, g, cob, anual, tramos, altitud = entorno
    monkeypatch.setattr(tablas, "FRANJA", 5)  # several strips, so accumulation is exercised
    huellas = tablas.construir(raiz, insumos, tmp_path / "t")
    assert set(huellas) == {"k0", "k2", "k3", "k5", "balance"}
    for nombre, (desfases, anios) in tablas.TABLAS.items():
        con_banda = nombre != "k0"
        esperado, balance = _esperado(g, cob, anual, tramos, altitud, insumos, desfases, anios,
                                      con_banda)
        obtenido = _leer(tmp_path / "t" / f"{nombre}.parquet", con_banda, len(desfases))
        assert obtenido.keys() == esperado.keys()
        for clave, (area, n) in esperado.items():
            assert obtenido[clave][0] == pytest.approx(area, rel=1e-9)
            assert obtenido[clave][1] == n
        if nombre == "k3":
            bal = pq.read_table(tmp_path / "t" / "balance.parquet").to_pydict()
            vistas = {
                (bal["celda"][i], bal["anio"][i], bal["grupo"][i], bal["tramo"][i],
                 bal["nivel1_previo"][i]): np.array([bal[k][i] for k in
                                                     ("w", "w_alt", "w_alt2", "w_dist",
                                                      "w_dist2")])
                for i in range(len(bal["celda"]))
            }
            assert vistas.keys() == balance.keys()
            for clave, v in balance.items():
                np.testing.assert_allclose(vistas[clave], v, rtol=1e-9)
    meta = pq.read_table(tmp_path / "t" / "k3.parquet").schema.metadata
    assert meta[b"huella"].decode() == huellas["k3"]


def test_k2_band_is_zero_after_the_last_interface_year(entorno, tmp_path):
    raiz, insumos, *_ = entorno
    tablas.construir(raiz, insumos, tmp_path / "t", ("k2",))
    t = pq.read_table(tmp_path / "t" / "k2.parquet").to_pydict()
    bandas_2022 = {b for a, b in zip(t["anio"], t["banda"], strict=True) if a == 2022}
    bandas_2021 = {b for a, b in zip(t["anio"], t["banda"], strict=True) if a == 2021}
    assert bandas_2022 == {0}
    assert bandas_2021 == {0, 1}  # 2021 reads the 2020 layer


def test_same_inputs_give_the_same_bits_whatever_the_strip_height(entorno, tmp_path,
                                                                  monkeypatch):
    raiz, insumos, *_ = entorno
    a = tablas.construir(raiz, insumos, tmp_path / "a", ("k3", "balance"))
    b = tablas.construir(raiz, insumos, tmp_path / "b", ("k3", "balance"))
    assert a == b
    monkeypatch.setattr(tablas, "FRANJA", 3)
    c = tablas.construir(raiz, insumos, tmp_path / "c", ("k3",))
    ta = pq.read_table(tmp_path / "a" / "k3.parquet")
    tc = pq.read_table(tmp_path / "c" / "k3.parquet")
    assert ta.column("n_px").to_pylist() == tc.column("n_px").to_pylist()
    np.testing.assert_allclose(ta.column("area_m2").to_numpy(), tc.column("area_m2").to_numpy(),
                               rtol=1e-9)
    assert set(c) == {"k3"}


def test_choices_are_checked_and_nothing_is_overwritten(entorno, tmp_path):
    raiz, insumos, *_ = entorno
    with pytest.raises(SourceError, match="unknown"):
        tablas.construir(raiz, insumos, tmp_path / "t", ("k4",))
    with pytest.raises(SourceError, match="together with k3"):
        tablas.construir(raiz, insumos, tmp_path / "t", ("balance",))
    tablas.construir(raiz, insumos, tmp_path / "t", ("k0",))
    with pytest.raises(SourceError, match="overwrite"):
        tablas.construir(raiz, insumos, tmp_path / "t", ("k0",))


def test_altitude_reader_checks_the_grid(entorno):
    _, insumos, g, *_, altitud = entorno
    assert (elevacion.leer_altitud(insumos.pendiente, g, 2, 3) == altitud[2:5]).all()
    otra = type(g)(g.oeste, g.norte, g.paso, g.ancho + 1, g.alto)
    with pytest.raises(SourceError, match="another grid"):
        elevacion.leer_altitud(insumos.pendiente, otra, 0, 1)


def test_command_line_builds_the_chosen_tables(entorno, tmp_path, capsys):
    raiz, insumos, *_ = entorno
    salida = tmp_path / "t"
    codigo = main(["estudio", "tablas", "--bruto", str(raiz), "--celdas", str(insumos.celdas),
                   "--pendiente", str(insumos.pendiente), "--interfaz", str(insumos.interfaz),
                   "--salida", str(salida), "--tablas", "k3", "balance"])
    assert codigo == 0
    texto = capsys.readouterr().out
    assert "k3:" in texto and "balance:" in texto
    assert sorted(p.name for p in salida.iterdir()) == ["balance.parquet", "k3.parquet"]


def _capa_lfs(carpeta, g, semilla=9):
    rng = np.random.default_rng(semilla)
    banderas = rng.integers(0, 4, (g.alto, g.ancho)).astype(np.uint8)
    carpeta.mkdir()
    (carpeta / "banderas.u8").write_bytes(banderas.tobytes())
    (carpeta / "lfs.json").write_text(json.dumps({"grilla": asdict(g)}), encoding="utf-8")
    return banderas


@pytest.mark.parametrize(("todos", "control"), [(2, 0), (0, 1), (2, 1)])
def test_scar_flags_leave_the_groups_they_name(entorno, tmp_path, todos, control):
    raiz, insumos, g, cob, anual, tramos, _altitud = entorno
    banderas = _capa_lfs(tmp_path / "lfs", g)
    con = tablas.Insumos(insumos.celdas, insumos.pendiente, insumos.interfaz,
                         tmp_path / "lfs", todos, control)
    tablas.construir(raiz, insumos, tmp_path / "sin", ("k3",))
    tablas.construir(raiz, con, tmp_path / "con", ("k3", "balance"))
    sin = pq.read_table(tmp_path / "sin" / "k3.parquet").to_pandas()
    t = pq.read_table(tmp_path / "con" / "k3.parquet")
    hecho = t.to_pandas()
    veces = (anual == 1).sum(axis=0)
    areas = g.area_filas_m2(0, ALTO)
    esperado_fuera = 0.0
    for anio in tablas.TABLAS["k3"][1]:
        b = fuego.ANIOS.index(anio)
        for f in range(ALTO):
            for c in range(ANCHO):
                if anual[0, f, c] == 255 or tramos[f, c] == elevacion.SIN_TRAMO:
                    continue
                codigos = [cob[cobertura.ANIOS_ESTUDIO.index(anio + d), f, c] for d in (-1, 2, 3)]
                if any(x in (0, 255, 27) for x in codigos):
                    continue
                grupo = 0 if veces[f, c] == 0 else (1 if anual[b, f, c] == 1 else -1)
                if veces[f, c] >= 2 and anual[b, f, c] == 1:
                    grupo = 2
                if grupo < 0:
                    continue
                if (banderas[f, c] & todos) or (grupo == 0 and banderas[f, c] & control):
                    esperado_fuera += areas[f]
    assert hecho.area_m2.sum() + esperado_fuera == pytest.approx(sin.area_m2.sum(), rel=1e-9)
    meta = t.schema.metadata
    assert float(meta[b"excluido_por_cicatrices_m2"]) == pytest.approx(esperado_fuera, rel=1e-9)
    assert meta[b"excluir_bits"] == str(todos).encode()
    assert meta[b"excluir_bits_control"] == str(control).encode()
    if not todos:
        quemado = hecho[hecho.grupo > 0].area_m2.sum()
        assert quemado == pytest.approx(sin[sin.grupo > 0].area_m2.sum(), rel=1e-12)
    bal = pq.read_table(tmp_path / "con" / "balance.parquet").schema.metadata
    assert bal[b"excluido_por_cicatrices_m2"] == meta[b"excluido_por_cicatrices_m2"]


def test_scar_exclusion_needs_the_layer(entorno):
    _, insumos, *_ = entorno
    with pytest.raises(SourceError, match="scar layer"):
        tablas.Insumos(insumos.celdas, insumos.pendiente, insumos.interfaz, None, 2, 0)
    with pytest.raises(SourceError, match="one byte"):
        tablas.Insumos(insumos.celdas, insumos.pendiente, insumos.interfaz, insumos.celdas, 256)


def test_left_out_areas_are_recorded(entorno, tmp_path):
    raiz, insumos, g, cob, anual, tramos, _altitud = entorno
    tablas.construir(raiz, insumos, tmp_path / "t", ("k3",))
    meta = pq.read_table(tmp_path / "t" / "k3.parquet").schema.metadata
    veces = (anual == 1).sum(axis=0)
    areas = g.area_filas_m2(0, ALTO)
    sin_tramo = codigo = 0.0
    for anio in tablas.TABLAS["k3"][1]:
        b = fuego.ANIOS.index(anio)
        for f in range(ALTO):
            for c in range(ANCHO):
                if anual[0, f, c] == 255:
                    continue
                if not (veces[f, c] == 0 or anual[b, f, c] == 1):
                    continue
                if tramos[f, c] == elevacion.SIN_TRAMO:
                    sin_tramo += areas[f]
                    continue
                codigos = [cob[cobertura.ANIOS_ESTUDIO.index(anio + d), f, c] for d in (-1, 2, 3)]
                if any(x in (0, 255, 27) for x in codigos):
                    codigo += areas[f]
    assert float(meta[b"excluido_sin_tramo_m2"]) == pytest.approx(sin_tramo, rel=1e-9)
    assert float(meta[b"excluido_codigo_no_valido_m2"]) == pytest.approx(codigo, rel=1e-9)
    assert codigo > 0 and sin_tramo > 0


def test_exploratory_table_marks_the_filter_window(entorno, tmp_path):
    raiz, insumos, g, cob, anual, tramos, _altitud = entorno
    tablas.construir(raiz, insumos, tmp_path / "t", ("u5",))
    t = pq.read_table(tmp_path / "t" / "u5.parquet").to_pydict()
    assert set(t["anio"]) <= set(range(2013, 2020))
    veces = (anual == 1).sum(axis=0)
    areas = g.area_filas_m2(0, ALTO)
    esperado = {}
    for anio in range(2013, 2020):
        b = fuego.ANIOS.index(anio)
        banda = interfaz.leer_banda(insumos.interfaz, g, anio - 1, 0, ALTO)
        for f in range(ALTO):
            for c in range(ANCHO):
                if anual[0, f, c] == 255 or tramos[f, c] == elevacion.SIN_TRAMO:
                    continue
                cods = [int(cob[cobertura.ANIOS_ESTUDIO.index(anio + d), f, c]) for d in (-1, 4, 5)]
                if any(x in (0, 255, 27) for x in cods):
                    continue
                if veces[f, c] == 0:
                    grupo = 0
                elif anual[b, f, c] == 1:
                    grupo = 1 if veces[f, c] == 1 else 2
                else:
                    continue
                marca = all(cob[cobertura.ANIOS_ESTUDIO.index(anio + d), f, c] == 24
                            for d in range(4))
                lat = float(g.latitudes(f, 1)[0])
                lon = float(g.longitudes(c, 1)[0])
                celda = int(h3n.cell_to_parent(h3n.latlng_to_cell(lat, lon, 8), 7))
                clave = (celda, anio, grupo, int(tramos[f, c]), int(banda[f, c]), int(marca), *cods)
                esperado[clave] = esperado.get(clave, 0.0) + areas[f]
    obtenido = {
        (t["celda"][i], t["anio"][i], t["grupo"][i], t["tramo"][i], t["banda"][i], t["m24"][i],
         t["c0"][i], t["c1"][i], t["c2"][i]): t["area_m2"][i]
        for i in range(len(t["celda"]))
    }
    assert obtenido.keys() == esperado.keys()
    for clave, area in esperado.items():
        assert obtenido[clave] == pytest.approx(area, rel=1e-9)
    assert 1 in set(t["m24"])


def test_lfs_table_groups_by_scar_count(entorno, tmp_path):
    raiz, insumos, g, cob, anual, tramos, _altitud = entorno
    rng = np.random.default_rng(11)
    conteo = rng.choice([0, 0, 1, 1, 2], size=(ALTO, ANCHO)).astype(np.uint8)
    temporada = np.where(conteo == 1, rng.integers(12, 19, (ALTO, ANCHO)), 0).astype(np.uint8)
    carpeta = tmp_path / "conteo"
    carpeta.mkdir()
    (carpeta / "conteo.u8").write_bytes(conteo.tobytes())
    (carpeta / "temporada_unica.u8").write_bytes(temporada.tobytes())
    (carpeta / "conteo.json").write_text(json.dumps({"grilla": asdict(g)}), encoding="utf-8")
    with pytest.raises(SourceError, match="scar count layer"):
        tablas.construir(raiz, insumos, tmp_path / "x", ("l5",))
    con = tablas.Insumos(insumos.celdas, insumos.pendiente, insumos.interfaz, conteo_lfs=carpeta)
    tablas.construir(raiz, con, tmp_path / "t", ("l5",))
    t = pq.read_table(tmp_path / "t" / "l5.parquet").to_pydict()
    veces = (anual == 1).sum(axis=0)
    areas = g.area_filas_m2(0, ALTO)
    esperado = {}
    for s in range(2014, 2019):
        banda = interfaz.leer_banda(insumos.interfaz, g, s - 2, 0, ALTO)
        for f in range(ALTO):
            for c in range(ANCHO):
                if anual[0, f, c] == 255 or tramos[f, c] == elevacion.SIN_TRAMO:
                    continue
                cods = [int(cob[cobertura.ANIOS_ESTUDIO.index(s + d), f, c]) for d in (-2, 4, 5)]
                if any(x in (0, 255, 27) for x in cods):
                    continue
                if conteo[f, c] == 0 and veces[f, c] == 0:
                    grupo = 0
                elif conteo[f, c] == 1 and temporada[f, c] == s - 2000:
                    grupo = 1
                else:
                    continue
                lat = float(g.latitudes(f, 1)[0])
                lon = float(g.longitudes(c, 1)[0])
                celda = int(h3n.cell_to_parent(h3n.latlng_to_cell(lat, lon, 8), 7))
                clave = (celda, s, grupo, int(tramos[f, c]), int(banda[f, c]), *cods)
                esperado[clave] = esperado.get(clave, 0.0) + areas[f]
    obtenido = {
        (t["celda"][i], t["anio"][i], t["grupo"][i], t["tramo"][i], t["banda"][i], t["c0"][i],
         t["c1"][i], t["c2"][i]): t["area_m2"][i]
        for i in range(len(t["celda"]))
    }
    assert obtenido.keys() == esperado.keys()
    for clave, area in esperado.items():
        assert obtenido[clave] == pytest.approx(area, rel=1e-9)
    assert {1, 0} <= set(t["grupo"]) and 2 not in set(t["grupo"])
