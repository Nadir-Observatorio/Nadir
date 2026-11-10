import json
import math
import zipfile
from datetime import date

import numpy as np
import pytest

from nadir import banco, exportacion
from nadir.__main__ import main
from nadir.capas import fuego, lfs
from nadir.estudio import validacion
from sinteticos import exportar
from test_lfs import FUENTE, cicatriz_tif, dbf

PASO = 0.002
OESTE, NORTE = -72.0, -33.0
ALTO, ANCHO = 40, 50
G = exportacion.Grilla(OESTE, NORTE, PASO, ANCHO, ALTO)


def agregar_fuego(zf, raiz, region, fid, umbral, fecha, mascara, oeste, norte, area_m2,
                  solapes=""):
    base = f"{raiz}/FireScar_{region}_ID{fid}_u{umbral}_{fecha}"
    ignicion = date(int(fecha[:4]), int(fecha[4:6]), int(fecha[6:]))
    zf.writestr(base + ".tif", cicatriz_tif(mascara, oeste, norte, paso=PASO))
    fila = {"FireID": fid, "Season": lfs.temporada(ignicion), "TotalArea": area_m2,
            "OverlapIDs": solapes}
    zf.writestr(base + ".dbf", dbf([fila]))


def _mascara(filas, columnas, f0, f1, c0, c1):
    m = np.zeros((filas, columnas), dtype=np.uint8)
    m[f0:f1, c0:c1] = 1
    return m


@pytest.fixture
def banco_completo(tmp_path):
    raiz = tmp_path / "bruto"
    mensual = np.zeros((13, ALTO, ANCHO), dtype=np.uint8)
    mensual[fuego.ANIOS.index(2017), 12:20, 14:24] = 2  # February 2017: season 2017
    mensual[fuego.ANIOS.index(2016), 20:22, 14:24] = 7  # July 2016: first month of season 2017
    mensual[fuego.ANIOS.index(2016), 30:34, 30:34] = 3  # March 2016: season 2016, not 2017
    anual = (mensual > 0).astype(np.uint8)
    car = raiz / fuego.CARPETA
    banco.preparar(car, fuego.FUENTE)
    exportar(car / "f.zip", fuego.PREFIJO_ANUAL, anual,
             tuple(fuego.banda_anual(a) for a in fuego.ANIOS), lado=16, paso=PASO)
    exportar(car / "f.zip", fuego.PREFIJO_MENSUAL, mensual,
             tuple(fuego.banda_mensual(a) for a in fuego.ANIOS), lado=16, paso=PASO)
    banco.registrar(car, "f.zip", "2013-2025", "earthengine:x", False)
    cl = raiz / lfs.CARPETA
    cl.mkdir(parents=True)
    banco.preparar(cl, FUENTE)
    ruta = "FireScar_CL-VS_Valparaiso_2009-2018.zip"
    with zipfile.ZipFile(cl / ruta, "w") as zf:
        r = "x/FireScar_CL-VS_Valparaiso_2017"
        # A: large scar of season 2017, rows 10-22, columns 12-22 of the grid
        agregar_fuego(zf, r, "CL-VS", 1, 400, "20170125", _mascara(12, 10, 0, 12, 0, 10),
                      OESTE + 12 * PASO, NORTE - 10 * PASO, 3_000_000.0)
        # Small neighbour of the same season overlapping the Fire burn to the east
        agregar_fuego(zf, r, "CL-VS", 2, 400, "20170201", _mascara(4, 3, 0, 4, 0, 3),
                      OESTE + 22 * PASO, NORTE - 12 * PASO, 400_000.0)
        # Small scar of the same season from another region inside the window: it leaves
        agregar_fuego(zf, r, "CL-CO", 6, 400, "20170115", _mascara(2, 2, 0, 2, 0, 2),
                      OESTE + 22 * PASO, NORTE - 20 * PASO, 100_000.0)
        # Small scar of another season inside the window: it stays in the comparison
        agregar_fuego(zf, r, "CL-VS", 5, 400, "20160110", _mascara(3, 2, 0, 3, 0, 2),
                      OESTE + 22 * PASO, NORTE - 17 * PASO, 300_000.0)
        # Large scar of another season and a large one outside the seven regions
        agregar_fuego(zf, r, "CL-VS", 3, 400, "20120110", _mascara(4, 4, 0, 4, 0, 4),
                      OESTE + 30 * PASO, NORTE - 30 * PASO, 3_000_000.0)
        agregar_fuego(zf, r, "CL-CO", 4, 400, "20170110", _mascara(4, 4, 0, 4, 0, 4),
                      OESTE + 2 * PASO, NORTE - 2 * PASO, 3_000_000.0, solapes="1")
    banco.registrar(cl, ruta, "p", lfs.PAGINA, False)
    return raiz, mensual


def test_jaccard_of_a_large_scar(banco_completo):
    raiz, mensual = banco_completo
    filas, resumen = validacion.chequeo_lfs(raiz)
    assert [f["id"] for f in filas] == [1]
    a = np.zeros((ALTO, ANCHO), dtype=bool)
    a[10:22, 12:22] = True
    vecina = np.zeros_like(a)
    vecina[12:16, 22:25] = True
    vecina[20:22, 22:24] = True  # same season, another region
    b = np.zeros_like(a)
    b[(mensual[fuego.ANIOS.index(2017)] >= 1) & (mensual[fuego.ANIOS.index(2017)] <= 6)] = True
    b[(mensual[fuego.ANIOS.index(2016)] >= 7)] = True
    paso_y = 1000.0 / (6371008.8 * math.radians(PASO))
    lat = NORTE - (np.nonzero(a)[0].mean() + 0.5) * PASO
    mx, my = math.ceil(paso_y / math.cos(math.radians(lat))), math.ceil(paso_y)
    caja = np.zeros_like(a)
    caja[10 - my:22 + my, 12 - mx:22 + mx] = True
    usar = caja & ~(vecina & ~a)
    w = np.broadcast_to(G.area_filas_m2(0, ALTO)[:, None], a.shape)
    esperado = w[a & b & usar].sum() / w[(a | b) & usar].sum()
    assert filas[0]["jaccard"] == round(float(esperado), 4)
    assert filas[0]["temporada"] == 2017 and filas[0]["solape"] is False
    assert filas[0]["area_total_ha"] == 300.0
    assert resumen["cicatrices"] == 1 and resumen["sin_solape_n"] == 1
    assert resumen["sin_solape_mediana"] == filas[0]["jaccard"]
    assert resumen["elegidas_sin_pixeles_en_la_grilla"] == 0
    assert resumen["jaccard_media"] == filas[0]["jaccard"]


def test_scar_iterator_matches_the_layer_rule(banco_completo):
    raiz, _ = banco_completo
    vistas = {cic.id: (f0, c0, v.sum()) for cic, f0, c0, v in lfs.recorrer(raiz, G)}
    assert vistas[1] == (10, 12, 120)
    assert vistas[2][2] == 12


def test_command_line_writes_table_and_summary(banco_completo, tmp_path, capsys):
    raiz, _ = banco_completo
    salida = tmp_path / "chequeo"
    assert main(["estudio", "chequeo-lfs", "--bruto", str(raiz), "--salida", str(salida)]) == 0
    assert "1 scars: mean Jaccard" in capsys.readouterr().out
    lineas = (salida / "jaccard_8_1.csv").read_text(encoding="utf-8").splitlines()
    assert lineas[0] == "id,region,temporada,area_lfs_ha,area_total_ha,solape,jaccard"
    assert json.loads((salida / "resumen_8_1.json").read_text("utf-8"))["referencia_media"] == 0.801
    assert main(["estudio", "chequeo-lfs", "--bruto", str(raiz), "--salida", str(salida)]) == 1


def test_scar_count_and_single_season(banco_completo, tmp_path):
    raiz, _ = banco_completo
    car = raiz / lfs.CARPETA
    origen = car / "FireScar_CL-VS_Valparaiso_2009-2018.zip"
    copia = car / "FireScar_CL-VS_copia.zip"
    with zipfile.ZipFile(origen) as zin, zipfile.ZipFile(copia, "w") as z:
        for info in zin.infolist():
            if "ID1_" in info.filename:  # the same scar repeated in another zip counts once
                z.writestr(info, zin.read(info))
        r = "y/FireScar_CL-VS_Valparaiso_2015"
        agregar_fuego(z, r, "CL-VS", 7, 400, "20150110", _mascara(2, 2, 0, 2, 0, 2),
                      OESTE + 12 * PASO, NORTE - 10 * PASO, 50_000.0)  # overlaps scar 1
        agregar_fuego(z, r, "CL-VS", 8, 400, "20200110", _mascara(2, 2, 0, 2, 0, 2),
                      OESTE + 40 * PASO, NORTE - 30 * PASO, 50_000.0)  # season 2020: ignored
    banco.registrar(car, "FireScar_CL-VS_copia.zip", "p", lfs.PAGINA, False)
    registro = lfs.construir_conteo(raiz, G, tmp_path / "conteo")
    conteo, temporada = lfs.leer_conteo(tmp_path / "conteo", G, 0, ALTO)
    assert conteo[10, 12] == 2 and temporada[10, 12] == 0  # scars 1 and 7
    assert conteo[15, 15] == 1 and temporada[15, 15] == 17  # scar 1 only, season 2017
    assert conteo[30, 30] == 1 and temporada[30, 30] == 12  # scar 3, season 2012
    assert conteo[31, 41] == 0  # the 2020 scar is past 2018
    assert registro["cicatrices"] == 7
    assert registro["pixeles_con_dos_o_mas"] == 4
    with pytest.raises(lfs.SourceError, match="overwrite"):
        lfs.construir_conteo(raiz, G, tmp_path / "conteo")
    otra = exportacion.Grilla(OESTE + PASO, NORTE, PASO, ANCHO, ALTO)
    with pytest.raises(lfs.SourceError, match="another grid"):
        lfs.leer_conteo(tmp_path / "conteo", otra, 0, 1)
