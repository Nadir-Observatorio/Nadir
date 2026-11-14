"""H5 of the interface module (descriptive, no contrast): exposure where burned forest does not
return.

Cells without return: resolution-6 H3 cells with their own effect (100 ha or more of comparable
burned forest) and a positive effect, from the split of H1 by cell (`mapas.por_celda`, the same
figures as the published maps).

Exposure dimensions, after Albarracín and Vera (2017):
- population: persons (`n_per`) and private dwellings (`n_vp`) of the 2024 Census, from urban and
  hamlet blocks (Manzanas) and rural entities (Entidades), shared among cells by area: a polygon
  gives each cell the share of its area that falls in the cell;
- infrastructure: school establishments (Mineduc 2021), health establishments in operation (DEIS)
  and fire companies (2017), counted by the cell that holds the point;
- productive systems: area of agriculture and forestry (level 1 class 3) of land cover in
  `ANIO_COBERTURA`;
- ecosystem: area of forest formation and natural non-forest formation (level 1 classes 1 and 2)
  of the same year.

Inside or outside the interface: the 2 km band of land cover code 24 of `ANIO_BANDA` (the year
before the land cover year, as in the estimator). A pixel is inside or outside by its own band; a
piece of a census polygon and an establishment by the band of the pixel that holds its centre or
its point. Every table reports areas and counts of aggregates: they do not describe the people or
the places inside a cell (ecological fallacy); figures of persons below `MIN_PERSONAS` are
suppressed with their dwellings.
"""

import csv
import hashlib
import io
import json
import tomllib
import zipfile
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import h3.api.numpy_int as h3n
import numpy as np
import numpy.typing as npt
import pandas as pd
import pyarrow.parquet as pq

from nadir import celdas, exportacion
from nadir.capas import cobertura, interfaz
from nadir.errors import SourceError
from nadir.estudio import mapas, regiones
from nadir.estudio.categorias import CATEGORIA, CATEGORIA_A_NIVEL1

RESOLUCION = 6
ANIO_COBERTURA = 2024
ANIO_BANDA = 2023
MIN_PERSONAS = 50
CAPAS_CENSO = ("Manzanas", "Entidades")
VARIABLES_CENSO = ("n_per", "n_vp")
CARPETAS_CENSO = tuple(
    Path("ine-censo2024") / n
    for n in (
        "cartografia",
        "cartografia-r05",
        "cartografia-r06",
        "cartografia-r07",
        "cartografia-r08",
        "cartografia-r09",
        "cartografia-r16",
    )
)
CARPETA_DEIS = Path("deis") / "establecimientos-salud"
CARPETA_IDE = Path("ide-chile") / "instalaciones-wfs"
ARCHIVO_EDUCACION = "establecimientos_educacion_escolar.geojson"
ARCHIVO_BOMBEROS = "companias_bomberos.geojson"
LADOS = ("dentro", "fuera", "sin_banda")
NIVEL1 = {"ecosistema_bosque": 1, "ecosistema_no_boscosa": 2, "productivo_agro_silvicola": 3}
NOTA = (
    "aggregates by H3 cell: they do not describe the persons or places inside a cell "
    "(ecological fallacy); persons below 50 are suppressed with their dwellings"
)

Banda = Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64]], npt.NDArray[np.uint8]]


# ---------------------------------------------------------------------------
# Bank
# ---------------------------------------------------------------------------


def registrados(carpeta: Path) -> list[str]:
    """Files of a bank folder whose size and sha256 match an entry of its provenance file.

    The provenance files of these sources were written by other projects of the bank, with
    extra fields; only `nombre`, `bytes` and `sha256` of each entry are read here. An entry
    without its file is skipped; a file that does not match its entry stops the reading.
    """
    ruta = carpeta / "procedencia.toml"
    if not ruta.exists():
        raise SourceError(f"{carpeta} has no provenance file")
    datos = tomllib.loads(ruta.read_text(encoding="utf-8"))
    vistos = []
    for entrada in datos.get("archivo", []):
        archivo = carpeta / str(entrada["nombre"])
        if not archivo.exists() or entrada["nombre"] in vistos:
            continue
        resumen = hashlib.sha256()
        with archivo.open("rb") as f:
            for bloque in iter(lambda: f.read(1 << 20), b""):
                resumen.update(bloque)
        if archivo.stat().st_size != entrada["bytes"] or resumen.hexdigest() != entrada["sha256"]:
            raise SourceError(f"{archivo} does not match its provenance entry")
        vistos.append(str(entrada["nombre"]))
    return vistos


# ---------------------------------------------------------------------------
# Cells and bands
# ---------------------------------------------------------------------------


def celdas_sin_retorno(tablas: Path) -> pd.DataFrame:
    """Resolution-6 cells with their own effect and a positive one."""
    t = mapas.por_celda(tablas)[RESOLUCION]
    t = t[t.efecto_pp.notna() & (t.efecto_pp > 0)]
    return t[["celda", "efecto_pp", "bosque_comparable_ha"]].reset_index(drop=True)


def lector_banda(carpeta: Path, grilla: exportacion.Grilla, anio: int) -> Banda:
    """Band of the pixel that holds each point: 0 inside, 1 outside, `interfaz.FUERA` elsewhere."""

    def banda(lat: npt.NDArray[np.float64], lon: npt.NDArray[np.float64]) -> npt.NDArray[np.uint8]:
        fila = np.floor((grilla.norte - lat) / grilla.paso).astype(np.int64)
        col = np.floor((lon - grilla.oeste) / grilla.paso).astype(np.int64)
        salida = np.full(lat.shape, interfaz.FUERA, dtype=np.uint8)
        ok = (fila >= 0) & (fila < grilla.alto) & (col >= 0) & (col < grilla.ancho)
        for f in np.unique(fila[ok]):
            en = ok & (fila == f)
            linea = interfaz.leer_banda(carpeta, grilla, anio, int(f), 1)[0]
            salida[en] = linea[col[en]]
        return salida

    return banda


def _lado(banda: npt.NDArray[np.uint8]) -> npt.NDArray[np.object_]:
    return np.where(banda == 0, "dentro", np.where(banda == 1, "fuera", "sin_banda")).astype(object)


# ---------------------------------------------------------------------------
# Population
# ---------------------------------------------------------------------------


def _parquet_de_zip(ruta: Path, capa: str) -> pd.DataFrame:
    with zipfile.ZipFile(ruta) as z:
        nombres = [n for n in z.namelist() if n.endswith(f"_{capa}.parquet")]
        if len(nombres) != 1:
            raise SourceError(f"{ruta}: expected one {capa} layer, found {len(nombres)}")
        datos = z.read(nombres[0])
    tabla = pq.read_table(io.BytesIO(datos), columns=[*VARIABLES_CENSO, "SHAPE"])
    return tabla.to_pandas()


def leer_censo(bruto: Path) -> pd.DataFrame:
    """Blocks and rural entities of the regional files, checked against the bank."""
    partes = []
    for carpeta in CARPETAS_CENSO:
        for nombre in registrados(bruto / carpeta):
            if not nombre.endswith(".zip"):
                continue
            for capa in CAPAS_CENSO:
                d = _parquet_de_zip(bruto / carpeta / nombre, capa)
                d["capa"] = capa
                partes.append(d)
    if not partes:
        raise SourceError("no census file recorded in the bank")
    censo = pd.concat(partes, ignore_index=True)
    if censo[list(VARIABLES_CENSO)].isna().any().any() or (censo.n_per < 0).any():
        raise SourceError("census counts with missing or negative values")
    return censo


def _hexagono(celda: int) -> object:
    from shapely.geometry import Polygon

    return Polygon([(lon, lat) for lat, lon in h3n.cell_to_boundary(celda)])


def poblacion(censo: pd.DataFrame, celdas_sel: Iterable[int], banda: Banda) -> pd.DataFrame:
    """Persons and dwellings by cell and side, each polygon shared by area.

    Areas are taken in geographic coordinates: the share of a polygon that falls in a cell is a
    ratio of two areas of the same polygon, whose scale changes by less than 0.1 % over 5 km.
    """
    from shapely import STRtree, area, centroid, from_wkb, get_x, get_y, intersection, make_valid

    sel = np.array(sorted(int(c) for c in celdas_sel), dtype=np.int64)
    hexes = [_hexagono(int(c)) for c in sel]
    arbol = STRtree(hexes)
    geoms = make_valid(from_wkb(censo.SHAPE.to_numpy()))
    i_pol, i_hex = arbol.query(geoms, predicate="intersects")
    if i_pol.size == 0:
        return pd.DataFrame(columns=["celda", "lado", *VARIABLES_CENSO])
    piezas = intersection(geoms[i_pol], np.array(hexes, dtype=object)[i_hex])
    total = area(geoms[i_pol])
    parte = np.divide(area(piezas), total, out=np.zeros(i_pol.size), where=total > 0)
    c = centroid(piezas)
    lado = _lado(banda(get_y(c), get_x(c)))
    d = pd.DataFrame({"celda": sel[i_hex], "lado": lado})
    for v in VARIABLES_CENSO:
        d[v] = censo[v].to_numpy(dtype=float)[i_pol] * parte
    d = d[parte > 0]
    return d.groupby(["celda", "lado"], as_index=False)[list(VARIABLES_CENSO)].sum()


# ---------------------------------------------------------------------------
# Establishments
# ---------------------------------------------------------------------------


def _puntos_geojson(ruta: Path) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    datos = json.loads(ruta.read_text(encoding="utf-8"))
    xy = np.array(
        [f["geometry"]["coordinates"][:2] for f in datos["features"] if f.get("geometry")],
        dtype=float,
    )
    return xy[:, 1], xy[:, 0]


def _puntos_deis(ruta: Path) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    with ruta.open(encoding="utf-8", newline="") as f:
        filas = list(csv.DictReader(f, delimiter=";"))
    lat, lon = [], []
    for r in filas:
        if not r["EstadoFuncionamiento"].startswith("Vigente") or r["FechaCierre"].strip():
            continue
        try:
            lat.append(float(r["Latitud"].replace(",", ".")))
            lon.append(float(r["Longitud"].replace(",", ".")))
        except ValueError:
            continue
    return np.array(lat), np.array(lon)


def leer_establecimientos(
    bruto: Path,
) -> dict[str, tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]:
    """Points of each kind of establishment, checked against the bank."""
    ide = set(registrados(bruto / CARPETA_IDE))
    deis = [n for n in registrados(bruto / CARPETA_DEIS) if n.endswith(".csv")]
    if ARCHIVO_EDUCACION not in ide or ARCHIVO_BOMBEROS not in ide or len(deis) != 1:
        raise SourceError("establishment files missing from the bank")
    return {
        "educacion": _puntos_geojson(bruto / CARPETA_IDE / ARCHIVO_EDUCACION),
        "salud": _puntos_deis(bruto / CARPETA_DEIS / deis[0]),
        "bomberos": _puntos_geojson(bruto / CARPETA_IDE / ARCHIVO_BOMBEROS),
    }


def establecimientos(
    puntos: dict[str, tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]],
    celdas_sel: Iterable[int],
    banda: Banda,
) -> pd.DataFrame:
    """Count of each kind by cell and side."""
    sel = {int(c) for c in celdas_sel}
    partes = []
    for tipo, (lat, lon) in puntos.items():
        celda = np.fromiter(
            (h3n.latlng_to_cell(a, b, RESOLUCION) for a, b in zip(lat, lon, strict=True)),
            dtype=np.int64,
            count=lat.size,
        )
        en = np.isin(celda, list(sel))
        d = pd.DataFrame({"celda": celda[en], "lado": _lado(banda(lat[en], lon[en])), "tipo": tipo})
        partes.append(d)
    todo = pd.concat(partes, ignore_index=True)
    return todo.groupby(["celda", "lado", "tipo"]).size().rename("n").reset_index()


# ---------------------------------------------------------------------------
# Land cover areas
# ---------------------------------------------------------------------------


def superficies(
    mosaico: exportacion.Mosaico,
    mapa: celdas.MapaCeldas,
    carpeta_interfaz: Path,
    celdas_sel: Iterable[int],
    filas_franja: int = 512,
) -> pd.DataFrame:
    """Area in hectares of each level 1 class by cell and side, land cover of `ANIO_COBERTURA`."""
    g = mosaico.grilla
    padres, posicion = mapa.padres(RESOLUCION)
    sel = np.isin(padres, np.array(sorted(int(c) for c in celdas_sel), dtype=np.int64))
    nombre = cobertura.banda(ANIO_COBERTURA)
    n_p = padres.size
    acumulado = np.zeros(n_p * 3 * 6)
    for f0 in range(0, g.alto, filas_franja):
        n = min(filas_franja, g.alto - f0)
        clase = mosaico.leer(nombre, f0, n)
        idx = mapa.indices(f0, n)
        ok = (clase != mosaico.sin_dato) & (idx != celdas.SIN_CELDA)
        pos = np.zeros(idx.shape, dtype=np.int64)
        pos[ok] = posicion[idx[ok].astype(np.int64)]
        ok &= sel[pos]
        if not ok.any():
            continue
        banda = interfaz.leer_banda(carpeta_interfaz, g, ANIO_BANDA, f0, n)
        lado = np.where(banda == 0, 0, np.where(banda == 1, 1, 2))
        nivel = CATEGORIA_A_NIVEL1[CATEGORIA[clase]].astype(np.int64)
        areas = np.broadcast_to(g.area_filas_m2(f0, n)[:, None], clase.shape)
        clave = (pos[ok] * 3 + lado[ok]) * 6 + nivel[ok]
        acumulado += np.bincount(clave, weights=areas[ok], minlength=acumulado.size)
    tabla = acumulado.reshape(n_p, 3, 6)
    filas = []
    for p in np.nonzero(sel)[0]:
        for j, lado_n in enumerate(LADOS):
            fila: dict[str, Any] = {"celda": int(padres[p]), "lado": lado_n}
            for col, nivel1 in NIVEL1.items():
                fila[f"{col}_ha"] = tabla[p, j, nivel1] / 1e4
            fila["total_ha"] = tabla[p, j].sum() / 1e4
            filas.append(fila)
    t = pd.DataFrame(filas)
    return t[t.total_ha > 0].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def _ancho(largo: pd.DataFrame, valores: list[str], columnas: str = "lado") -> pd.DataFrame:
    t = largo.pivot_table(index="celda", columns=columnas, values=valores, aggfunc="sum")
    t.columns = [f"{v}_{c}" for v, c in t.columns]
    return t


def suprimir(t: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Blank persons below `MIN_PERSONAS` (and above zero) with their dwellings."""
    t = t.copy()
    n = 0
    for lado in LADOS:
        p, v = f"n_per_{lado}", f"n_vp_{lado}"
        if p not in t:
            continue
        bajo = (t[p] > 0) & (t[p] < MIN_PERSONAS)
        n += int(bajo.sum())
        t.loc[bajo, [p, v]] = np.nan
    return t, n


def tablas_h5(
    sin_retorno: pd.DataFrame,
    pob: pd.DataFrame,
    est: pd.DataFrame,
    sup: pd.DataFrame,
    region: dict[int, str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Table by cell and by region (before suppression in the region sums)."""
    base = sin_retorno.set_index("celda")
    partes = [base]
    if not pob.empty:
        partes.append(_ancho(pob, list(VARIABLES_CENSO)))
    if not est.empty:
        e = est.assign(clave=est.tipo + "_" + est.lado).pivot_table(
            index="celda", columns="clave", values="n", aggfunc="sum"
        )
        partes.append(e)
    if not sup.empty:
        partes.append(_ancho(sup, [f"{c}_ha" for c in NIVEL1]))
    t = pd.concat(partes, axis=1).reindex(base.index)
    tipos = ("educacion", "salud", "bomberos")
    esperadas = [f"{v}_{lado}" for v in VARIABLES_CENSO for lado in LADOS[:2]]
    esperadas += [f"{tipo}_{lado}" for tipo in tipos for lado in LADOS[:2]]
    esperadas += [f"{c}_ha_{lado}" for c in NIVEL1 for lado in LADOS[:2]]
    extra = [c for c in t.columns if c not in esperadas and c not in base.columns]
    t = t.reindex(columns=[*base.columns, *esperadas, *sorted(extra)])
    numericas = [c for c in t.columns if c not in ("efecto_pp", "bosque_comparable_ha")]
    t[numericas] = t[numericas].fillna(0.0)
    t.insert(0, "region", [region[int(c)] for c in t.index])
    por_region = t.drop(columns=["efecto_pp"]).groupby("region").sum()
    por_region.insert(0, "celdas", t.groupby("region").size())
    celda_sup, n_celda = suprimir(t)
    region_sup, n_region = suprimir(por_region)
    celda_sup = celda_sup.reset_index()
    celda_sup.insert(0, "h3", [h3n.int_to_str(int(c)) for c in celda_sup.celda])
    celda_sup = celda_sup.drop(columns="celda")
    conteo = {"suprimidas_celda": n_celda, "suprimidas_region": n_region}
    return celda_sup, region_sup.reset_index(), conteo


def escribir(
    por_celda: pd.DataFrame, por_region: pd.DataFrame, meta: dict[str, object], carpeta: Path
) -> list[Path]:
    """CSV by cell and by region and the JSON record, LF line ends. Never overwrites."""
    rutas = [carpeta / f"exposicion_celdas_r{RESOLUCION}.csv", carpeta / "exposicion_regiones.csv",
             carpeta / "exposicion.json"]
    for r in rutas:
        if r.exists():
            raise SourceError(f"refusing to overwrite {r}")
    carpeta.mkdir(parents=True, exist_ok=True)
    por_celda.to_csv(rutas[0], index=False, float_format="%.1f", lineterminator="\n")
    por_region.to_csv(rutas[1], index=False, float_format="%.1f", lineterminator="\n")
    texto = json.dumps(meta, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    rutas[2].write_bytes(texto.encode("utf-8"))
    return rutas


def analizar_h5(
    tablas: Path, bruto: Path, carpeta_celdas: Path, carpeta_interfaz: Path, limites: Path
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """H5 tables and their record."""
    sin_retorno = celdas_sin_retorno(tablas)
    sel = sin_retorno.celda.to_numpy()
    with cobertura.abrir(bruto) as mosaico:
        g = mosaico.grilla
        mapa = celdas.abrir(carpeta_celdas, g)
        sup = superficies(mosaico, mapa, carpeta_interfaz, sel)
    banda = lector_banda(carpeta_interfaz, g, ANIO_BANDA)
    pob = poblacion(leer_censo(bruto), sel, banda)
    est = establecimientos(leer_establecimientos(bruto), sel, banda)
    region, _ = regiones.asignar(sel, limites)
    por_celda, por_region, conteo = tablas_h5(sin_retorno, pob, est, sup, region)
    meta: dict[str, object] = {
        "celdas_sin_retorno": int(sel.size),
        "resolucion": RESOLUCION,
        "anio_cobertura": ANIO_COBERTURA,
        "anio_banda_interfaz": ANIO_BANDA,
        "min_personas": MIN_PERSONAS,
        **conteo,
        "regla_region": "region of the centre of the resolution-6 cell",
        "nota": NOTA,
    }
    return por_celda, por_region, meta
