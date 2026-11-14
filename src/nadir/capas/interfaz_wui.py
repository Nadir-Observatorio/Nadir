"""Wildland-urban interface of Chile (Zenodo, 2025) on the native grid of the MapBiomas exports.

Source: "Multi-threshold Wildland Urban Interface of Chile" (Gelabert Vadillo, Félez and
Rodrigues, 2025; doi:10.5281/zenodo.17670820; CC BY 4.0). Two rasters of 30 m in EPSG:32719,
`WuInterface_intensity.tif` and `WuIntermix_intensity.tif`: each pixel holds the share of the 12
threshold combinations that classify it as Interface (or Intermix); pixels never classified are
no data. A pixel is interface here when either intensity is greater than `UMBRAL` (a majority of
the combinations); no data is not interface.

Each pixel of the export grid takes the source pixel that holds its centre (nearest neighbour,
GDAL warp in strips with an exact transformation, tolerance 0); nothing is averaged. The layer
is fixed in time: the same band serves every fire year.

One file, one byte per pixel, row-major, on the export grid: `banda_wui.u8` with 0 = interface,
1 = not interface, `FUERA` where the land cover export of `ANIO_MASCARA` has no data.
"""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import numpy.typing as npt
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject

from nadir import banco, exportacion
from nadir.capas import cobertura
from nadir.errors import SourceError

CARPETA = Path("zenodo") / "wui-chile-2025"
ARCHIVOS = ("WuInterface_intensity.tif", "WuIntermix_intensity.tif")
UMBRAL = 0.5
FUERA = 255
ANIO_MASCARA = 2020
FILAS_FRANJA = 512
CRS_EXPORTACION = "EPSG:4326"
_BANDA = "banda_wui.u8"
_REGISTRO = "registro_wui.json"


def fuentes(bruto: Path) -> dict[str, banco.Archivo]:
    """The two recorded rasters, checked against their size and sha256 in the bank."""
    registrados = {a.nombre: a for a in banco.comprobar(bruto / CARPETA)}
    faltan = [n for n in ARCHIVOS if n not in registrados]
    if faltan:
        raise SourceError(f"not recorded in {bruto / CARPETA}: {faltan}")
    return {n: registrados[n] for n in ARCHIVOS}


def clasificar(
    interfaz: npt.NDArray[np.float32], intermix: npt.NDArray[np.float32]
) -> npt.NDArray[np.uint8]:
    """0 where either intensity is greater than `UMBRAL`, else 1 (no data is not interface)."""
    con_nan = np.errstate(invalid="ignore")
    with con_nan:
        dentro = (interfaz > UMBRAL) | (intermix > UMBRAL)
    return np.where(dentro, 0, 1).astype(np.uint8)


def _muestrear(
    fuente: rasterio.io.DatasetReader, grilla: exportacion.Grilla, fila: int, filas: int
) -> npt.NDArray[np.float32]:
    destino = np.full((filas, grilla.ancho), np.nan, dtype=np.float32)
    reproject(
        source=rasterio.band(fuente, 1),
        destination=destino,
        src_nodata=fuente.nodata,
        dst_transform=from_origin(
            grilla.oeste, grilla.norte - fila * grilla.paso, grilla.paso, grilla.paso
        ),
        dst_crs=CRS_EXPORTACION,
        dst_nodata=np.nan,
        resampling=Resampling.nearest,
        tolerance=0.0,
    )
    return destino


def construir(bruto: Path, mosaico: exportacion.Mosaico, carpeta: Path) -> dict[str, object]:
    """Interface band of the Zenodo layer for every pixel of the land cover export.

    Writes into `carpeta` and never overwrites. Returns the record also saved there.
    """
    if (carpeta / _BANDA).exists() or (carpeta / _REGISTRO).exists():
        raise SourceError(f"refusing to overwrite the WUI files in {carpeta}")
    registrados = fuentes(bruto)
    carpeta.mkdir(parents=True, exist_ok=True)
    g = mosaico.grilla
    nombre_mascara = cobertura.banda(ANIO_MASCARA)
    resumen = hashlib.sha256()
    dentro = interfaz_px = sin_dato_px = 0
    area_interfaz = 0.0
    parcial = carpeta / f"{_BANDA}.part"
    with (
        rasterio.open(bruto / CARPETA / ARCHIVOS[0]) as f_interfaz,
        rasterio.open(bruto / CARPETA / ARCHIVOS[1]) as f_intermix,
    ):
        if f_interfaz.crs != f_intermix.crs or f_interfaz.transform != f_intermix.transform:
            raise SourceError("the two WUI rasters are not on the same grid")
        with parcial.open("wb") as salida:
            for f0 in range(0, g.alto, FILAS_FRANJA):
                n = min(FILAS_FRANJA, g.alto - f0)
                a = _muestrear(f_interfaz, g, f0, n)
                b = _muestrear(f_intermix, g, f0, n)
                banda = clasificar(a, b)
                fuera = mosaico.leer(nombre_mascara, f0, n) == mosaico.sin_dato
                banda[fuera] = FUERA
                usar = ~fuera
                dentro += int(usar.sum())
                interfaz_px += int((banda == 0).sum())
                sin_dato_px += int((usar & np.isnan(a) & np.isnan(b)).sum())
                area_interfaz += float(
                    ((banda == 0) * g.area_filas_m2(f0, n)[:, None]).sum()
                )
                datos = banda.tobytes()
                salida.write(datos)
                resumen.update(datos)
        crs_fuente = str(f_interfaz.crs)
        resolucion = [float(v) for v in f_interfaz.res]
    parcial.replace(carpeta / _BANDA)
    registro: dict[str, object] = {
        "grilla": asdict(g),
        "metodo": "nearest neighbour: each export pixel takes the WUI pixel that holds its "
        "centre (GDAL warp, exact transformation); interface when Interface or Intermix "
        f"intensity is greater than {UMBRAL}; no data is not interface",
        "umbral": UMBRAL,
        "anio_mascara": ANIO_MASCARA,
        "crs_fuente": crs_fuente,
        "resolucion_fuente_m": resolucion,
        "fuentes": {n: {"bytes": a.bytes, "sha256": a.sha256} for n, a in registrados.items()},
        "pixeles_dentro": dentro,
        "pixeles_interfaz": interfaz_px,
        "pixeles_sin_dato_fuente": sin_dato_px,
        "area_interfaz_ha": round(area_interfaz / 1e4, 1),
        "sha256_banda": resumen.hexdigest(),
    }
    texto = json.dumps(registro, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    (carpeta / _REGISTRO).write_bytes(texto.encode("utf-8"))
    return registro


def leer_banda(
    carpeta: Path, grilla: exportacion.Grilla, fila: int, filas: int
) -> npt.NDArray[np.uint8]:
    """WUI band for rows `fila` to `fila + filas - 1`."""
    registro = json.loads((carpeta / _REGISTRO).read_text(encoding="utf-8"))
    if not exportacion.misma_grilla(exportacion.Grilla(**registro["grilla"]), grilla):
        raise SourceError(f"the WUI band in {carpeta} belongs to another grid")
    with (carpeta / _BANDA).open("rb") as origen:
        origen.seek(fila * grilla.ancho)
        datos = np.fromfile(origen, dtype="u1", count=filas * grilla.ancho)
    if datos.size != filas * grilla.ancho:
        raise SourceError(f"{carpeta / _BANDA} is shorter than its grid")
    return datos.reshape(filas, grilla.ancho)
