"""Validation of MapBiomas Fire against the 2024 Viña del Mar fire.

Two pieces, on the 20 m grid of the published severity map of that fire (its `clases.tif`, from
the repository incendio-vina-2024), restricted to its four municipalities:

- Agreement with the severity map, both ways: the share of each severity class that Fire maps as
  burned in February 2024, and how the area Fire maps as burned splits across severity classes.
- Accuracy against the Copernicus EMS delineation EMSR715 (AOI01, version 2), with the sampling
  design of that study: 300 pixels per predicted stratum, a minimum Chebyshev distance of 10
  pixels, seed 20240202, estimates weighted by stratum area (Olofsson et al. 2014).

Each Fire value is the monthly product's pixel that holds the centre of the 20 m pixel.

Functions marked as taken from that repository are copied unchanged in their logic, so both
studies read the same references in the same way.
"""

import hashlib
import io
import json
import math
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import rasterio
from affine import Affine

from nadir import exportacion
from nadir.capas import fuego, lfs
from nadir.errors import SourceError

SEMILLA = 20240202
MUESTRA_POR_ESTRATO = 300
DISTANCIA_MINIMA_PX = 10
FEBRERO = 2
ANIO = 2024
Z95 = 1.959963984540054
HUELLA_CLASES = "ef09b76717fc2855e0fcaf1cd939ecf2e69e6ea7ea416c9f4618e050244f6902"
EMSR715_BYTES = 178149313
EMSR715_SHA256 = "aa31440615508ba60cba1fe3da0c9e980d308d6547d76a7a0bf21a81b0a25905"
EMSR715_PRODUCTO = "EMSR715_AOI01_DEL_PRODUCT"
COMUNAS = ("05101", "05109", "05801", "05804")
NOMBRES_CLASE = {
    1: "Enhanced regrowth, high",
    2: "Enhanced regrowth, low",
    3: "Unburned",
    4: "Low severity",
    5: "Moderate-low severity",
    6: "Moderate-high severity",
    7: "High severity",
}
_BLOQUE = 1 << 20


def huella_contenido(ruta: Path) -> str:
    """sha256 of band 1 content (dtype, shape, CRS, transform and pixels), independent of the
    file compression. Taken from the repository incendio-vina-2024."""
    with rasterio.open(ruta) as r:
        datos = r.read(1)
        crs = r.crs.to_string() if r.crs else ""
        transform = tuple(round(float(v), 9) for v in tuple(r.transform)[:6])
    if np.issubdtype(datos.dtype, np.floating):
        datos = np.where(np.isnan(datos), np.array(np.nan, dtype=datos.dtype), datos)
    datos = np.ascontiguousarray(datos.astype(datos.dtype.newbyteorder("<")))
    h = hashlib.sha256()
    h.update(f"{datos.dtype.str}|{datos.shape}|{crs}|{transform}|".encode("ascii"))
    h.update(datos.tobytes())
    return h.hexdigest()


def comprobar_archivo(ruta: Path, tamano: int, sha256: str) -> None:
    """Raise unless `ruta` is the published file (size and sha256)."""
    if not ruta.is_file():
        raise SourceError(f"{ruta} not found")
    if ruta.stat().st_size != tamano:
        raise SourceError(f"{ruta.name}: {ruta.stat().st_size} bytes, expected {tamano}")
    h = hashlib.sha256()
    with ruta.open("rb") as origen:
        while bloque := origen.read(_BLOQUE):
            h.update(bloque)
    if h.hexdigest() != sha256:
        raise SourceError(f"{ruta.name}: sha256 {h.hexdigest()[:12]}, expected {sha256[:12]}")


def leer_clases(ruta: Path) -> tuple[npt.NDArray[np.uint8], Affine, str]:
    """Severity classes, their transform and CRS, after checking the content hash."""
    huella = huella_contenido(ruta)
    if huella != HUELLA_CLASES:
        raise SourceError(f"{ruta.name}: content hash {huella[:12]}, expected {HUELLA_CLASES[:12]}")
    with rasterio.open(ruta) as r:
        return r.read(1), r.transform, r.crs.to_string()


def leer_comunas(ruta: Path) -> dict[str, Any]:
    """The four municipalities as a GeoJSON FeatureCollection (EPSG:4326)."""
    datos: dict[str, Any] = json.loads(ruta.read_text(encoding="utf-8"))
    codigos = {f["properties"]["CUT_COM"] for f in datos.get("features", [])}
    if codigos != set(COMUNAS):
        raise SourceError(f"{ruta.name} has municipalities {sorted(codigos)}, expected {COMUNAS}")
    return datos


def mascara_zonas(
    geojson: dict[str, Any], crs: str, transform: Affine, forma: tuple[int, int]
) -> npt.NDArray[np.bool_]:
    """Pixels whose centre falls in any feature of a GeoJSON FeatureCollection (EPSG:4326).
    Taken from the repository incendio-vina-2024."""
    from pyproj import Transformer
    from rasterio.features import rasterize
    from shapely.geometry import shape
    from shapely.ops import transform as reproyectar

    a_crs = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform
    formas = [(reproyectar(a_crs, shape(f["geometry"])), 1) for f in geojson["features"]]
    if not formas:
        raise SourceError("the zone collection is empty")
    mascara: npt.NDArray[np.bool_] = rasterize(
        formas, out_shape=forma, transform=transform, fill=0, dtype="uint8"
    ).astype(bool)
    return mascara


def _bytes_internos(paquete: Path, nombre: str) -> bytes:
    with zipfile.ZipFile(paquete) as z:
        copias = {}
        for info in z.infolist():
            if info.filename == nombre:
                datos = z.read(info)
                copias[hashlib.sha256(datos).hexdigest()] = datos
    if not copias:
        raise SourceError(f"{nombre} is not in {paquete.name}")
    if len(copias) > 1:
        raise SourceError(f"{nombre} appears with different contents")
    return next(iter(copias.values()))


def delineacion_emsr715(paquete: Path, crs: str) -> tuple[list[Any], list[Any]]:
    """(burned polygons, footprint of the post-event image) of the EMSR715 AOI01 delineation,
    version 2, reprojected to `crs`. Taken from the repository incendio-vina-2024."""
    from pyproj import Transformer
    from shapely.geometry import shape
    from shapely.ops import transform as reproyectar

    producto, version = EMSR715_PRODUCTO, "v2"
    interno = io.BytesIO(_bytes_internos(paquete, f"{producto}_{version}.zip"))
    with zipfile.ZipFile(interno) as z:
        quemado = json.loads(z.read(f"{producto}_observedEventA_{version}.json"))
        huella = json.loads(z.read(f"{producto}_imageFootprintA_{version}.json"))
    a_crs = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform
    poligonos = [reproyectar(a_crs, shape(f["geometry"])) for f in quemado["features"]]
    imagen = [
        reproyectar(a_crs, shape(f["geometry"]))
        for f in huella["features"]
        if str(f["properties"]["or_src_id"]) == "2"
    ]
    if not poligonos or not imagen:
        raise SourceError(f"{producto}_{version}: no burned polygons or no post-event footprint")
    return poligonos, imagen


def _muestrear_estrato(
    candidatos: npt.NDArray[np.intp],
    n: int,
    ancho: int,
    distancia_px: int,
    rng: np.random.Generator,
) -> npt.NDArray[np.intp]:
    elegidos: list[int] = []
    filas: list[int] = []
    columnas: list[int] = []
    for i in rng.permutation(candidatos):
        r, c = divmod(int(i), ancho)
        if distancia_px > 0 and filas:
            cerca = (np.abs(np.asarray(filas) - r) < distancia_px) & (
                np.abs(np.asarray(columnas) - c) < distancia_px
            )
            if cerca.any():
                continue
        elegidos.append(int(i))
        filas.append(r)
        columnas.append(c)
        if len(elegidos) == n:
            break
    return np.asarray(elegidos, dtype=np.intp)


def muestrear_estratos(
    prediccion: npt.NDArray[np.bool_],
    valido: npt.NDArray[np.bool_],
    n_por_estrato: int,
    semilla: int,
    distancia_px: int = 0,
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    """Flat indices of the (burned, unburned) strata: one generator, burned stratum first, and a
    candidate rejected when a kept pixel is closer than `distancia_px` in both rows and columns.
    Taken from the repository incendio-vina-2024."""
    if prediccion.shape != valido.shape or prediccion.ndim != 2:
        raise SourceError("prediction and validity must be 2-D arrays of the same shape")
    if n_por_estrato < 1 or distancia_px < 0:
        raise SourceError("the sample size must be positive and the distance not negative")
    rng = np.random.default_rng(semilla)
    salida = []
    for nombre, estrato in (("burned", prediccion & valido), ("unburned", ~prediccion & valido)):
        elegidos = _muestrear_estrato(
            np.flatnonzero(estrato), n_por_estrato, prediccion.shape[1], distancia_px, rng
        )
        if elegidos.size < n_por_estrato:
            raise SourceError(
                f"{nombre} stratum: {elegidos.size} of {n_por_estrato} points with a minimum "
                f"distance of {distancia_px} px"
            )
        salida.append(elegidos)
    return salida[0], salida[1]


def fuego_en_centros(
    bruto: Path, transform: Affine, crs: str, forma: tuple[int, int], anio: int = ANIO
) -> npt.NDArray[np.uint8]:
    """Monthly Fire value of `anio` at the centre of each pixel of a projected grid."""
    from pyproj import Transformer

    alto, ancho = forma
    filas, columnas = np.mgrid[0:alto, 0:ancho]
    x = transform.c + (columnas + 0.5) * transform.a
    y = transform.f + (filas + 0.5) * transform.e
    lon, lat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(x, y)
    anual, mensual = fuego.abrir(bruto)
    anual.cerrar()
    with mensual:
        g = mensual.grilla
        fila_f = np.floor((g.norte - lat) / g.paso).astype(np.int64)
        col_f = np.floor((lon - g.oeste) / g.paso).astype(np.int64)
        f0, f1 = int(fila_f.min()), int(fila_f.max())
        ventana = mensual.leer(fuego.banda_mensual(anio), f0, f1 - f0 + 1)
    valores: npt.NDArray[np.uint8] = ventana[fila_f - f0, col_f]
    return valores


def validar(bruto: Path, clases_tif: Path, comunas: Path, emsr715: Path) -> dict[str, Any]:
    """Agreement with the severity map and accuracy against EMSR715, as a dictionary."""
    from rasterio.features import rasterize
    from shapely.ops import unary_union

    comprobar_archivo(emsr715, EMSR715_BYTES, EMSR715_SHA256)
    clases, transform, crs = leer_clases(clases_tif)
    en_comunas = mascara_zonas(leer_comunas(comunas), crs, transform, clases.shape)
    valor = fuego_en_centros(bruto, transform, crs, clases.shape)
    quemado_fuego = valor == FEBRERO
    fuera_regiones = int(((valor == fuego.FUERA) & en_comunas).sum())
    ha_px = abs(transform.a * transform.e) / 10_000.0
    salida: dict[str, Any] = {
        "clases_huella": HUELLA_CLASES,
        "fuego_255_en_comunas_px": fuera_regiones,
    }
    salida["8.2"] = _concordancia(clases, en_comunas, quemado_fuego, ha_px)

    quemado, huella = delineacion_emsr715(emsr715, crs)
    referencia = rasterize(
        [(p, 1) for p in quemado], out_shape=clases.shape, transform=transform, fill=0,
        dtype="uint8",
    ).astype(bool)
    en_huella = rasterize(
        [(unary_union(huella), 1)], out_shape=clases.shape, transform=transform, fill=0,
        dtype="uint8",
    ).astype(bool)
    dominio = (clases > 0) & en_comunas & en_huella
    salida["8.3"] = _exactitud(quemado_fuego, referencia, dominio, ha_px)
    return salida


def _concordancia(
    clases: npt.NDArray[np.uint8],
    en_comunas: npt.NDArray[np.bool_],
    quemado_fuego: npt.NDArray[np.bool_],
    ha_px: float,
) -> dict[str, Any]:
    dominio = en_comunas & (clases > 0)
    salida: dict[str, Any] = {
        "dominio_ha": round(dominio.sum() * ha_px, 1),
        "comunas_ha": round(en_comunas.sum() * ha_px, 1),
        "clase0_en_comunas_ha": round((en_comunas & (clases == 0)).sum() * ha_px, 1),
        "a_por_clase": {},
    }
    for codigo, nombre in NOMBRES_CLASE.items():
        m = dominio & (clases == codigo)
        n = int(m.sum())
        q = int((m & quemado_fuego).sum())
        salida["a_por_clase"][f"{codigo} {nombre}"] = {
            "ha": round(n * ha_px, 1),
            "fuego_quemado_ha": round(q * ha_px, 1),
            "pct": round(100 * q / n, 1) if n else None,
        }
    fq = en_comunas & quemado_fuego
    total_fq = int(fq.sum())
    b: dict[str, Any] = {"fuego_quemado_en_comunas_ha": round(total_fq * ha_px, 1)}
    for etiqueta, sel in (
        ("clase 0 (sin dato)", clases == 0),
        ("clases 1 a 3", (clases >= 1) & (clases <= 3)),
        ("clase 4 (baja)", clases == 4),
        ("clases 5 a 7", clases >= 5),
    ):
        k = int((fq & sel).sum())
        b[etiqueta] = {"ha": round(k * ha_px, 1), "pct": round(100 * k / total_fq, 1)}
    salida["b_del_area_fuego"] = b
    return salida


def _exactitud(
    pred: npt.NDArray[np.bool_],
    referencia: npt.NDArray[np.bool_],
    dominio: npt.NDArray[np.bool_],
    ha_px: float,
) -> dict[str, Any]:
    bq, bn = muestrear_estratos(pred, dominio, MUESTRA_POR_ESTRATO, SEMILLA, DISTANCIA_MINIMA_PX)
    r = referencia.ravel()
    n = {"q": bq.size, "n": bn.size}
    nij = {
        ("q", "q"): int(r[bq].sum()),
        ("q", "n"): int(bq.size - r[bq].sum()),
        ("n", "q"): int(r[bn].sum()),
        ("n", "n"): int(bn.size - r[bn].sum()),
    }
    area = {"q": float((pred & dominio).sum()), "n": float((~pred & dominio).sum())}
    total = area["q"] + area["n"]
    peso = {k: v / total for k, v in area.items()}
    p = {(i, j): peso[i] * nij[(i, j)] / n[i] for i in "qn" for j in "qn"}
    oa = p[("q", "q")] + p[("n", "n")]
    u = {i: nij[(i, i)] / n[i] for i in "qn"}
    var_oa = sum(peso[i] ** 2 * u[i] * (1 - u[i]) / (n[i] - 1) for i in "qn")
    pj = p[("q", "q")] + p[("n", "q")]
    se_pj = np.sqrt(
        sum(
            peso[i] ** 2 * (nij[(i, "q")] / n[i]) * (1 - nij[(i, "q")] / n[i]) / (n[i] - 1)
            for i in "qn"
        )
    )
    prod = p[("q", "q")] / pj
    nq, nn = area["q"], area["n"]
    nj = nq * nij[("q", "q")] / n["q"] + nn * nij[("n", "q")] / n["n"]
    var_prod = (1 / nj**2) * (
        nq**2 * (1 - prod) ** 2 * u["q"] * (1 - u["q"]) / (n["q"] - 1)
        + prod**2
        * nn**2
        * (nij[("n", "q")] / n["n"])
        * (1 - nij[("n", "q")] / n["n"])
        / (n["n"] - 1)
    )

    def ic(v: float, se: float) -> list[float]:
        return [round(100 * (v - Z95 * se), 1), round(100 * (v + Z95 * se), 1)]

    return {
        "dominio_ha": round(total * ha_px, 1),
        "fuego_quemado_ha": round(area["q"] * ha_px, 1),
        "emsr715_quemado_en_dominio_ha": round((referencia & dominio).sum() * ha_px, 1),
        "matriz_muestra": {
            "fuego_q_ref_q": nij[("q", "q")],
            "fuego_q_ref_n": nij[("q", "n")],
            "fuego_n_ref_q": nij[("n", "q")],
            "fuego_n_ref_n": nij[("n", "n")],
        },
        "exactitud_global_pct": round(100 * oa, 1),
        "exactitud_global_ic95": ic(oa, np.sqrt(var_oa)),
        "comision_pct": round(100 * (1 - u["q"]), 1),
        "comision_ic95": ic(1 - u["q"], np.sqrt(u["q"] * (1 - u["q"]) / (n["q"] - 1))),
        "omision_pct": round(100 * (1 - prod), 1),
        "omision_ic95": ic(1 - prod, np.sqrt(var_prod)),
        "area_quemada_ajustada_ha": round(pj * total * ha_px, 1),
        "area_quemada_ajustada_ic95_ha": [
            round((pj - Z95 * se_pj) * total * ha_px, 1),
            round((pj + Z95 * se_pj) * total * ha_px, 1),
        ],
    }


# Check against the Landscape Fire Scars Database for Chile (not independent: it shares authors
# with MapBiomas Fire, whose own assessment reported a mean Jaccard of 0.801 on 100 scars).
REGIONES_LFS = ("CL-VS", "CL-RM", "CL-LI", "CL-ML", "CL-BI", "CL-AR")
TEMPORADAS_LFS = (2014, 2015, 2016, 2017, 2018)
AREA_MINIMA_LFS_HA = 250.0
MARGEN_LFS_M = 1000.0
RADIO_TIERRA_M = 6371008.8


Huellas = dict[int, tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]]


def chequeo_lfs(bruto: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Jaccard index of each scar of the 2014-2018 seasons above 250 ha against Fire.

    The Fire season s joins the monthly product's July to December of year s - 1 and January to
    June of year s. For each scar A, B is the Fire burned area in the same season inside the
    scar's bounding box widened by 1 km on each side; pixels of other scars of the same season,
    of any size and any region, that are not in A leave both A-or-B and A-and-B. Areas are
    weighted by row. Scars chosen but without pixels on the grid are counted in the summary.
    """
    anual, mensual = fuego.abrir(bruto)
    anual.cerrar()
    with mensual:
        g = mensual.grilla
        huellas: Huellas = {}
        temporadas: dict[int, int] = {}
        elegidas: list[lfs.Cicatriz] = []
        sin_pixeles = 0
        for cic, f0, c0, ventana in lfs.recorrer(bruto, g):
            if cic.temporada not in TEMPORADAS_LFS:
                continue
            elegible = (
                cic.region in REGIONES_LFS and cic.area_total_m2 / 10_000 > AREA_MINIMA_LFS_HA
            )
            ff, cc = np.nonzero(ventana)
            if ff.size == 0:
                sin_pixeles += int(elegible)
                continue
            huellas[cic.id] = ((ff + f0).astype(np.int64), (cc + c0).astype(np.int64))
            temporadas[cic.id] = cic.temporada
            if elegible:
                elegidas.append(cic)
        filas = [_jaccard_de(mensual, huellas, temporadas, c) for c in sorted(elegidas, key=_id)]
    resumen = _resumen_lfs(filas)
    resumen["elegidas_sin_pixeles_en_la_grilla"] = sin_pixeles
    return filas, resumen


def _id(cicatriz: lfs.Cicatriz) -> int:
    return int(cicatriz.id)


def _jaccard_de(
    mensual: exportacion.Mosaico, huellas: Huellas, temporadas: dict[int, int], cic: lfs.Cicatriz
) -> dict[str, Any]:
    g = mensual.grilla
    area_fila = g.area_filas_m2(0, g.alto)
    paso_y = MARGEN_LFS_M / (RADIO_TIERRA_M * math.radians(g.paso))
    s = cic.temporada
    rr, cc = huellas[cic.id]
    lat_c = g.norte - (rr.mean() + 0.5) * g.paso
    mx = math.ceil(paso_y / math.cos(math.radians(lat_c)))
    my = math.ceil(paso_y)
    f0, f1 = max(0, int(rr.min()) - my), min(g.alto, int(rr.max()) + my + 1)
    c0, c1 = max(0, int(cc.min()) - mx), min(g.ancho, int(cc.max()) + mx + 1)
    a = np.zeros((f1 - f0, c1 - c0), dtype=bool)
    a[rr - f0, cc - c0] = True
    otras = np.zeros_like(a)
    for oid, (orr, occ) in huellas.items():
        if oid == cic.id or temporadas[oid] != s:
            continue
        if orr.max() < f0 or orr.min() >= f1 or occ.max() < c0 or occ.min() >= c1:
            continue
        ok = (orr >= f0) & (orr < f1) & (occ >= c0) & (occ < c1)
        otras[orr[ok] - f0, occ[ok] - c0] = True
    despues = mensual.leer(fuego.banda_mensual(s), f0, f1 - f0)[:, c0:c1]
    quemado = (despues >= 1) & (despues <= 6)
    if s - 1 >= fuego.ANIOS[0]:
        antes = mensual.leer(fuego.banda_mensual(s - 1), f0, f1 - f0)[:, c0:c1]
        quemado |= (antes >= 7) & (antes <= 12)
    usar = ~(otras & ~a)
    w = np.broadcast_to(area_fila[f0:f1, None], a.shape)
    interseccion = float(w[a & quemado & usar].sum())
    union = float(w[(a | quemado) & usar].sum())
    return {
        "id": int(cic.id),
        "region": cic.region,
        "temporada": s,
        "area_lfs_ha": round(float(w[a].sum()) / 1e4, 2),
        "area_total_ha": round(cic.area_total_m2 / 1e4, 2),
        "solape": bool(cic.solapes.strip()),
        "jaccard": round(interseccion / union, 4) if union > 0 else None,
    }


def _resumen_lfs(filas: list[dict[str, Any]]) -> dict[str, Any]:
    j = np.array([f["jaccard"] for f in filas if f["jaccard"] is not None], dtype=float)
    js = np.array(
        [f["jaccard"] for f in filas if f["jaccard"] is not None and not f["solape"]], dtype=float
    )
    return {
        "cicatrices": len(filas),
        "jaccard_media": round(float(j.mean()), 4) if j.size else None,
        "jaccard_mediana": round(float(np.median(j)), 4) if j.size else None,
        "sin_solape_n": int(js.size),
        "sin_solape_media": round(float(js.mean()), 4) if js.size else None,
        "sin_solape_mediana": round(float(np.median(js)), 4) if js.size else None,
        "referencia_media": 0.801,
    }
