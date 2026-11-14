"""National context box (amendment V3-E2): descriptive, no effect and no contrast.

For every region of Chile:
- area affected by wildfires according to CONAF (summary of occurrence and damage by
  municipality), added up over the fire seasons that match the fire years of the study, 2012-2013
  to 2020-2021; Ñuble was created from Biobío in 2018, so the two go together;
- area of wildland-urban interface in the layer of amendment V3-E1 (Interface or Intermix
  intensity greater than 0.5) and the share of the national interface area in each region. The
  share is of the national interface, not of the region's land: OpenStreetMap regions include the
  territorial sea, so they are not a land denominator.

Regional outlines come from the OpenStreetMap extract of Chile in the bank (administrative level
4). The effect of fire is measured only in the seven regions of MapBiomas Fire; this table only
places them in the country.
"""

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nadir.errors import SourceError

TEMPORADAS = tuple(f"{a}-{a + 1}" for a in range(2012, 2021))
COLUMNA_TOTAL = "TOTAL SUPERFICIE AFECTADA"
UMBRAL_WUI = 0.5
FILAS_FRANJA = 1024
# CONAF roman numerals to ISO 3166-2 codes; Biobío and Ñuble share one row.
ISO = {
    "XV": "CL-AP", "I": "CL-TA", "II": "CL-AN", "III": "CL-AT", "IV": "CL-CO", "V": "CL-VS",
    "RM": "CL-RM", "XIII": "CL-RM", "VI": "CL-LI", "VII": "CL-ML", "VIII": "CL-BI",
    "XVI": "CL-BI", "IX": "CL-AR", "XIV": "CL-LR", "X": "CL-LL", "XI": "CL-AI", "XII": "CL-MA",
}
NOMBRES = {
    "CL-AP": "Arica y Parinacota", "CL-TA": "Tarapacá", "CL-AN": "Antofagasta",
    "CL-AT": "Atacama", "CL-CO": "Coquimbo", "CL-VS": "Valparaíso", "CL-RM": "Metropolitana",
    "CL-LI": "O'Higgins", "CL-ML": "Maule", "CL-BI": "Biobío y Ñuble", "CL-AR": "La Araucanía",
    "CL-LR": "Los Ríos", "CL-LL": "Los Lagos", "CL-AI": "Aysén", "CL-MA": "Magallanes",
}
ESTUDIO = {"CL-VS", "CL-RM", "CL-LI", "CL-ML", "CL-BI", "CL-AR"}
# Provinces that identify a block of the summary whose region cell is blank.
PROVINCIAS = {"CONCEPCION": "VIII", "CONCEPCIÓN": "VIII", "ARAUCO": "VIII", "BIOBIO": "VIII",
              "BIO BIO": "VIII", "BIOBÍO": "VIII", "BÍO BÍO": "VIII", "BIO-BIO": "VIII",
              "DIGUILLIN": "XVI", "ITATA": "XVI", "PUNILLA": "XVI"}


def _nombre_hoja(nombres: list[str], temporada: str) -> str:
    inicio, fin = temporada.split("-")
    for n in nombres:
        limpio = n.replace(" ", "")
        if limpio == f"{inicio}-{fin}":
            return n
    raise SourceError(f"no sheet for season {temporada}")


def leer_conaf(ruta: Path) -> pd.DataFrame:
    """Area affected (ha) by region and season; each season adds up to its national total."""
    import openpyxl

    libro = openpyxl.load_workbook(ruta, read_only=True, data_only=True)
    filas_salida = []
    for temporada in TEMPORADAS:
        hoja = libro[_nombre_hoja(libro.sheetnames, temporada)]
        filas = list(hoja.iter_rows(values_only=True))
        cab = next(i for i, f in enumerate(filas[:10]) if f and f[0] == "REGION")
        columna = next(j for j, v in enumerate(filas[cab]) if v and COLUMNA_TOTAL in str(v))
        codigo: str | None = None
        nacional = None
        por_region: dict[str, float] = {}
        for f in filas[cab + 1 :]:
            primero = str(f[0]).strip() if f[0] is not None else ""
            if primero == "TOTAL NACIONAL":
                nacional = float(f[columna] or 0)
                break
            if primero == "TOTAL":
                if codigo is None:
                    raise SourceError(f"{temporada}: a regional total with no region")
                iso = ISO[codigo]
                por_region[iso] = por_region.get(iso, 0.0) + float(f[columna] or 0)
                codigo = None
                continue
            if codigo is None:
                if primero in ISO:
                    codigo = primero
                elif primero == "" and f[1] is not None:
                    provincia = str(f[1]).strip().upper()
                    if provincia not in PROVINCIAS:
                        raise SourceError(f"{temporada}: block without region ({provincia})")
                    codigo = PROVINCIAS[provincia]
        if nacional is None:
            raise SourceError(f"{temporada}: no national total")
        if not math.isclose(sum(por_region.values()), nacional, rel_tol=1e-9, abs_tol=1e-6):
            raise SourceError(f"{temporada}: regions add up to {sum(por_region.values())}, "
                              f"not {nacional}")
        filas_salida += [{"temporada": temporada, "iso": k, "ha": v} for k, v in por_region.items()]
    return pd.DataFrame(filas_salida)


def regiones_osm(ruta_pbf: Path) -> dict[str, Any]:
    """Regions of Chile (administrative level 4) of an OpenStreetMap extract, as GeoJSON."""
    import osmium
    import shapely.wkb
    from shapely.geometry import mapping

    fabrica = osmium.geom.WKBFactory()
    rasgos: list[dict[str, Any]] = []

    class _Lector(osmium.SimpleHandler):
        def area(self, a: osmium.osm.Area) -> None:
            t = a.tags
            if (t.get("boundary") == "administrative" and t.get("admin_level") == "4"
                    and not a.from_way() and str(t.get("ISO3166-2", "")).startswith("CL-")):
                g = shapely.wkb.loads(fabrica.create_multipolygon(a), hex=True)
                rasgos.append({"type": "Feature", "geometry": mapping(g),
                               "properties": {"iso": t.get("ISO3166-2"), "nombre": t.get("name")}})

    _Lector().apply_file(str(ruta_pbf), locations=True, idx="flex_mem")
    rasgos.sort(key=lambda r: str(r["properties"]["iso"]))
    return {"type": "FeatureCollection", "features": rasgos}


def interfaz_por_region(
    interfaz: Path, intermix: Path, regiones: dict[str, Any]
) -> pd.DataFrame:
    """Interface area of each region, on the grid of the WUI layer."""
    import rasterio
    from pyproj import Transformer
    from rasterio.features import rasterize
    from rasterio.windows import Window
    from shapely.geometry import shape
    from shapely.ops import transform as transformar

    isos = sorted({str(r["properties"]["iso"]) for r in regiones["features"]})
    indice = {iso: i + 1 for i, iso in enumerate(isos)}
    wui = np.zeros(len(isos) + 1)
    with rasterio.open(interfaz) as a, rasterio.open(intermix) as b:
        if a.crs != b.crs or a.transform != b.transform or a.shape != b.shape:
            raise SourceError("the two WUI rasters are not on the same grid")
        proy = Transformer.from_crs("EPSG:4326", a.crs, always_xy=True).transform
        formas = [(transformar(proy, shape(r["geometry"])), indice[str(r["properties"]["iso"])])
                  for r in regiones["features"]]
        celda_ha = abs(a.transform.a * a.transform.e) / 1e4
        for f0 in range(0, a.height, FILAS_FRANJA):
            n = min(FILAS_FRANJA, a.height - f0)
            ventana = Window(0, f0, a.width, n)
            etiqueta = rasterize(formas, out_shape=(n, a.width),
                                 transform=a.window_transform(ventana), fill=0, dtype="uint8")
            if not etiqueta.any():
                continue
            x = a.read(1, window=ventana)
            y = b.read(1, window=ventana)
            with np.errstate(invalid="ignore"):
                dentro = (x > UMBRAL_WUI) | (y > UMBRAL_WUI)
            wui += np.bincount(etiqueta[dentro], minlength=wui.size) * celda_ha
    return pd.DataFrame({"iso": isos, "interfaz_ha": wui[1:]})


def tabla(conaf: pd.DataFrame, wui: pd.DataFrame) -> pd.DataFrame:
    """One row per region (Biobío with Ñuble), north to south, with the study regions marked."""
    quemado = conaf.groupby("iso").ha.sum().rename("afectada_ha_2013_2021")
    w = wui.assign(iso=wui.iso.replace({"CL-NB": "CL-BI"})).groupby("iso")[["interfaz_ha"]].sum()
    t = w.join(quemado, how="outer").fillna(0.0)
    t["interfaz_pct_nacional"] = 100 * t.interfaz_ha / t.interfaz_ha.sum()
    t["afectada_pct_nacional"] = 100 * t.afectada_ha_2013_2021 / t.afectada_ha_2013_2021.sum()
    t["estudio"] = t.index.isin(sorted(ESTUDIO))
    orden = list(NOMBRES)
    t = t.reindex([i for i in orden if i in t.index])
    t.insert(0, "region", [NOMBRES[i] for i in t.index])
    return t.reset_index()


def escribir(t: pd.DataFrame, meta: dict[str, object], carpeta: Path) -> list[Path]:
    """CSV of the table and JSON of its record, LF line ends. Never overwrites."""
    rutas = [carpeta / "contexto_nacional.csv", carpeta / "contexto_nacional.json"]
    for r in rutas:
        if r.exists():
            raise SourceError(f"refusing to overwrite {r}")
    carpeta.mkdir(parents=True, exist_ok=True)
    t.to_csv(rutas[0], index=False, float_format="%.2f", lineterminator="\n")
    texto = json.dumps(meta, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    rutas[1].write_bytes(texto.encode("utf-8"))
    return rutas


CARPETA_CONAF = Path("conaf") / "estadisticas-ocurrencia-dano"
CARPETA_OSM = Path("osm") / "geofabrik-chile"


def construir(bruto: Path) -> tuple[pd.DataFrame, dict[str, Any], dict[str, object]]:
    """Table, regional outlines and record, from the files checked against the bank."""
    from nadir import banco
    from nadir.capas import interfaz_wui
    from nadir.estudio.exposicion import registrados

    xlsx = [a for a in banco.comprobar(bruto / CARPETA_CONAF) if a.nombre.endswith(".xlsx")]
    pbf = [n for n in registrados(bruto / CARPETA_OSM) if n.endswith(".osm.pbf")]
    if len(xlsx) != 1 or len(pbf) != 1:
        raise SourceError("expected one CONAF workbook and one OpenStreetMap extract in the bank")
    fuentes = interfaz_wui.fuentes(bruto)
    regiones = regiones_osm(bruto / CARPETA_OSM / pbf[0])
    conaf = leer_conaf(bruto / CARPETA_CONAF / xlsx[0].nombre)
    carpeta_wui = bruto / interfaz_wui.CARPETA
    wui = interfaz_por_region(
        carpeta_wui / interfaz_wui.ARCHIVOS[0], carpeta_wui / interfaz_wui.ARCHIVOS[1], regiones
    )
    t = tabla(conaf, wui)
    meta: dict[str, object] = {
        "temporadas": list(TEMPORADAS),
        "umbral_wui": UMBRAL_WUI,
        "regiones_osm": len(regiones["features"]),
        "fuentes": {
            "conaf": {"archivo": xlsx[0].nombre, "sha256": xlsx[0].sha256},
            "osm": pbf[0],
            "wui": {n: a.sha256 for n, a in fuentes.items()},
        },
        "nota": "descriptive: no effect outside the seven study regions; Biobío includes Ñuble; "
        "the interface share is of the national interface area",
    }
    return t, regiones, meta
