"""Copernicus DEM GLO-30: elevation tiles of 1 x 1 degree, from the public archive on AWS.

Copernicus GLO-30 is a digital surface model (it includes vegetation and buildings) at one arc
second, about 30 m, in EPSG:4326. It is published as Cloud Optimized GeoTIFFs, one per degree,
named after the south-west corner (`S36_00_W073_00` covers 36° S to 35° S and 73° W to 72° W).
Tiles that are only ocean do not exist.

The tiles needed are those that hold at least one pixel of the land cover export, so elevation
covers exactly the study area. They are downloaded as published, verified and recorded in the
bank; nothing is reprojected or resampled here.
"""

import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import numpy.typing as npt
import rasterio
from rasterio.windows import Window

from nadir import banco, exportacion
from nadir.area import ACHATAMIENTO, SEMIEJE_MAYOR
from nadir.capas import cobertura
from nadir.errors import SourceError

CARPETA = Path("copernicus") / "dem-glo30"
BASE = "https://copernicus-dem-30m.s3.amazonaws.com"
FRANJA = 2048
PERIODO = "2010-2015"
"""Acquisition years of the radar data behind GLO-30 (December 2010 to January 2015)."""

FUENTE = banco.Fuente(
    organismo="ESA, programa Copernicus (archivo público en AWS Open Data)",
    conjunto="Copernicus DEM GLO-30 (DSM, 1 segundo de arco), teselas de 1 grado",
    fuente_pagina="https://registry.opendata.aws/copernicus-dem/",
    licencia="Copernicus-DEM-GLO-30",
    atribucion=(
        "produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and "
        "Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all "
        "rights reserved"
    ),
)


def nombre(latitud_sur: int, longitud_oeste: int) -> str:
    """Name of the tile whose south-west corner is at these whole degrees."""
    if not -90 <= latitud_sur < 90 or not -180 <= longitud_oeste < 180:
        raise SourceError(f"no tile at ({latitud_sur}, {longitud_oeste})")
    ns = f"{'S' if latitud_sur < 0 else 'N'}{abs(latitud_sur):02d}_00"
    eo = f"{'W' if longitud_oeste < 0 else 'E'}{abs(longitud_oeste):03d}_00"
    return f"Copernicus_DSM_COG_10_{ns}_{eo}_DEM"


def url(tesela: str) -> str:
    """Download address of one tile."""
    return f"{BASE}/{tesela}/{tesela}.tif"


def teselas(mosaico: exportacion.Mosaico, banda: str) -> list[str]:
    """Tiles that hold at least one pixel of `banda` that is not the export's no-data value."""
    vistas: set[tuple[int, int]] = set()
    g = mosaico.grilla
    for fila, datos in mosaico.franjas(banda, FRANJA):
        filas = np.flatnonzero(np.any(datos != mosaico.sin_dato, axis=1))
        if filas.size == 0:
            continue
        latitudes = g.latitudes(fila, datos.shape[0])
        longitudes = g.longitudes(0, datos.shape[1])
        for f in filas.tolist():
            columnas = np.flatnonzero(datos[f] != mosaico.sin_dato)
            lat = math.floor(latitudes[f])
            for lon in np.unique(np.floor(longitudes[columnas])).astype(int).tolist():
                vistas.add((lat, lon))
    return sorted(nombre(lat, lon) for lat, lon in vistas)


def descargar(bruto: Path, nombres: list[str]) -> tuple[list[banco.Archivo], list[str]]:
    """Download the tiles not yet in the bank, verified and recorded.

    Returns the new records and the tiles the archive does not have (403 or 404, which on this
    bucket means the object does not exist: an ocean-only tile). Any other answer stops.
    """
    carpeta = bruto / CARPETA
    banco.preparar(carpeta, FUENTE)
    ya = {a.nombre for a in banco.comprobar(carpeta)}
    nuevos: list[banco.Archivo] = []
    ausentes: list[str] = []
    for tesela in nombres:
        archivo = f"{tesela}.tif"
        if archivo in ya:
            continue
        estado, _ = banco.estado_remoto(url(tesela))
        if estado in (403, 404):
            ausentes.append(tesela)
            continue
        if estado != 200:
            raise SourceError(f"{url(tesela)}: HTTP {estado}")
        nuevos.append(banco.descargar(url(tesela), carpeta, archivo, PERIODO, True))
    return nuevos, ausentes


def necesarias(bruto: Path) -> list[str]:
    """Tiles that the land cover export of the bank needs."""
    with cobertura.abrir(bruto) as mosaico:
        return teselas(mosaico, cobertura.banda(cobertura.ANIOS_ESTUDIO[-1]))


# ---------------------------------------------------------------------------------------------
# Slope and altitude on the land cover grid
# ---------------------------------------------------------------------------------------------

ARCO = 3600
"""Pixels per degree of GLO-30 at these latitudes (one arc second); pixel centres sit on whole
arc seconds, so the tiles form one global grid."""

CORTES = (10.0, 25.0)
"""Slope bands in degrees: below 10, 10 to below 25, 25 or more."""

SIN_TRAMO = 255
SIN_ALTITUD = -32768
_TRAMOS = "tramos_pendiente.u8"
_ALTITUD = "altitud_m.i16"
_REGISTRO = "pendiente.json"


def _abrir_teselas(bruto: Path) -> dict[tuple[int, int], Path]:
    """Recorded tiles by their top-left pixel index on the global arc-second grid."""
    carpeta = bruto / CARPETA
    teselas_banco: dict[tuple[int, int], Path] = {}
    for archivo in banco.comprobar(carpeta):
        ruta = carpeta / archivo.nombre
        with rasterio.open(ruta) as datos:
            if datos.crs is None or datos.crs.to_string() != "EPSG:4326":
                raise SourceError(f"{archivo.nombre}: reference system {datos.crs}")
            if datos.width != ARCO or datos.height != ARCO or datos.count != 1:
                raise SourceError(f"{archivo.nombre}: expected one band of {ARCO} x {ARCO}")
            t = datos.transform
            if abs(t.a - 1 / ARCO) > 1e-12 or abs(t.e + 1 / ARCO) > 1e-12 or t.b or t.d:
                raise SourceError(f"{archivo.nombre}: cells are not one arc second")
            # The transform gives the outer corner; pixel centres sit half a pixel inside.
            fila = round(-(t.f + t.e / 2) * ARCO)
            columna = round((t.c + t.a / 2) * ARCO)
            teselas_banco[(fila, columna)] = ruta
    if not teselas_banco:
        raise SourceError(f"no elevation tiles recorded in {carpeta}")
    return teselas_banco


def _ventana_dem(
    teselas_banco: dict[tuple[int, int], Path], fila: int, filas: int, columna: int, columnas: int
) -> npt.NDArray[np.float64]:
    """Elevation on the global arc-second grid for rows and columns given as global indices.

    Rows count southward from the equator, columns eastward from the meridian. Where no tile
    exists the value is NaN.
    """
    salida = np.full((filas, columnas), np.nan, dtype=np.float64)
    for (f0, c0), ruta in teselas_banco.items():
        arriba, abajo = max(fila, f0), min(fila + filas, f0 + ARCO)
        izquierda, derecha = max(columna, c0), min(columna + columnas, c0 + ARCO)
        if arriba >= abajo or izquierda >= derecha:
            continue
        with rasterio.open(ruta) as datos:
            ventana = Window(izquierda - c0, arriba - f0, derecha - izquierda, abajo - arriba)
            salida[arriba - fila : abajo - fila, izquierda - columna : derecha - columna] = (
                datos.read(1, window=ventana).astype(np.float64)
            )
    return salida


def pendiente_horn(dem: npt.NDArray[np.float64], fila_global: int) -> npt.NDArray[np.float64]:
    """Slope in degrees (Horn, 3 x 3) of the inner pixels of `dem`, which carries a 1-pixel halo.

    `fila_global` is the global row index of the first row of `dem`. Spacing follows latitude on
    the WGS84 ellipsoid: meridian radius for north-south, prime-vertical radius times cos(lat)
    for east-west. Any missing neighbour gives NaN.
    """
    filas = dem.shape[0] - 2
    latitudes = -(np.arange(fila_global + 1, fila_global + 1 + filas, dtype=np.float64)) / ARCO
    seno = np.sin(np.radians(latitudes))
    e2 = ACHATAMIENTO * (2 - ACHATAMIENTO)
    meridiano = SEMIEJE_MAYOR * (1 - e2) / (1 - e2 * seno**2) ** 1.5
    normal = SEMIEJE_MAYOR / np.sqrt(1 - e2 * seno**2)
    paso = math.radians(1 / ARCO)
    dy = (meridiano * paso)[:, None]
    dx = (normal * np.cos(np.radians(latitudes)) * paso)[:, None]
    a, b, c = dem[:-2, :-2], dem[:-2, 1:-1], dem[:-2, 2:]
    d, f = dem[1:-1, :-2], dem[1:-1, 2:]
    g, h, i = dem[2:, :-2], dem[2:, 1:-1], dem[2:, 2:]
    dzdx = ((c + 2 * f + i) - (a + 2 * d + g)) / (8 * dx)
    dzdy = ((g + 2 * h + i) - (a + 2 * b + c)) / (8 * dy)
    resultado: npt.NDArray[np.float64] = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
    return resultado


def tramo(pendiente: npt.NDArray[np.float64]) -> npt.NDArray[np.uint8]:
    """Slope band 0, 1 or 2 by `CORTES`; `SIN_TRAMO` where the slope is missing."""
    bandas = np.digitize(pendiente, CORTES).astype(np.uint8)
    bandas[~np.isfinite(pendiente)] = SIN_TRAMO
    return bandas


def construir_pendiente(
    bruto: Path, mosaico: exportacion.Mosaico, banda: str, carpeta: Path, alto: int = 512
) -> dict[str, object]:
    """Slope band and altitude for every pixel of an export, saved in `carpeta`.

    Slope is computed on the native elevation grid, and each export pixel takes the value of the
    elevation pixel that holds its centre, without averaging. Pixels outside the export (its
    no-data in `banda`) get `SIN_TRAMO` and `SIN_ALTITUD`. Never overwrites.
    """
    if (carpeta / _REGISTRO).exists() or (carpeta / _TRAMOS).exists():
        raise SourceError(f"refusing to overwrite the slope files in {carpeta}")
    teselas_banco = _abrir_teselas(bruto)
    carpeta.mkdir(parents=True, exist_ok=True)
    g = mosaico.grilla
    columnas_dem = np.rint(g.longitudes(0, g.ancho) * ARCO).astype(np.int64)
    c_min, c_max = int(columnas_dem.min()) - 1, int(columnas_dem.max()) + 1
    cuenta = np.zeros(3, dtype=np.int64)
    dentro_sin_dem = 0
    partes = {_TRAMOS: carpeta / f"{_TRAMOS}.part", _ALTITUD: carpeta / f"{_ALTITUD}.part"}
    with partes[_TRAMOS].open("wb") as s_tramo, partes[_ALTITUD].open("wb") as s_alt:
        for fila, datos in mosaico.franjas(banda, alto):
            filas_dem = np.rint(-g.latitudes(fila, datos.shape[0]) * ARCO).astype(np.int64)
            f_min, f_max = int(filas_dem.min()) - 1, int(filas_dem.max()) + 1
            dem = _ventana_dem(teselas_banco, f_min, f_max - f_min + 1, c_min, c_max - c_min + 1)
            pendiente = pendiente_horn(dem, f_min)
            r = (filas_dem - f_min - 1)[:, None]
            c = (columnas_dem - c_min - 1)[None, :]
            valores = pendiente[r, c]
            altura = dem[r + 1, c + 1]
            dentro = datos != mosaico.sin_dato
            bandas = tramo(valores)
            bandas[~dentro] = SIN_TRAMO
            if np.any(np.abs(altura[np.isfinite(altura)]) > 32_000):
                raise SourceError("elevation outside the 16-bit range kept for altitude")
            alt = np.where(np.isfinite(altura), np.rint(altura), SIN_ALTITUD)
            alt[~dentro] = SIN_ALTITUD
            dentro_sin_dem += int(np.sum(dentro & (bandas == SIN_TRAMO)))
            cuenta += np.bincount(bandas[bandas != SIN_TRAMO], minlength=3)[:3]
            s_tramo.write(bandas.astype("u1").tobytes())
            s_alt.write(alt.astype("<i2").tobytes())
    for nombre, parcial in partes.items():
        parcial.replace(carpeta / nombre)
    registro: dict[str, object] = {
        "grilla": asdict(g),
        "banda_de_area": banda,
        "metodo": "Horn 3x3 on the native GLO-30 grid; export pixel takes the elevation pixel "
        "holding its centre",
        "cortes_grados": list(CORTES),
        "pixeles_por_tramo": [int(x) for x in cuenta],
        "pixeles_dentro_sin_dem": dentro_sin_dem,
        "teselas_sha256": {a.nombre: a.sha256 for a in banco.comprobar(bruto / CARPETA)},
        "sha256": {n: _sha256(carpeta / n) for n in (_TRAMOS, _ALTITUD)},
    }
    texto = json.dumps(registro, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    (carpeta / _REGISTRO).write_bytes(texto.encode("utf-8"))
    return registro


def _sha256(ruta: Path) -> str:
    resumen = hashlib.sha256()
    with ruta.open("rb") as origen:
        while bloque := origen.read(1 << 20):
            resumen.update(bloque)
    return resumen.hexdigest()


def leer_tramos(
    carpeta: Path, grilla: exportacion.Grilla, fila: int, filas: int
) -> npt.NDArray[np.uint8]:
    """Slope bands of rows `fila` to `fila + filas - 1`, after checking the saved grid."""
    registro = json.loads((carpeta / _REGISTRO).read_text(encoding="utf-8"))
    if registro.get("grilla") != asdict(grilla):
        raise SourceError(f"the slope files in {carpeta} belong to another grid")
    with (carpeta / _TRAMOS).open("rb") as origen:
        origen.seek(fila * grilla.ancho)
        datos = np.fromfile(origen, dtype="u1", count=filas * grilla.ancho)
    if datos.size != filas * grilla.ancho:
        raise SourceError(f"{carpeta / _TRAMOS} is shorter than its grid")
    return datos.reshape(filas, grilla.ancho)


def leer_altitud(
    carpeta: Path, grilla: exportacion.Grilla, fila: int, filas: int
) -> npt.NDArray[np.int16]:
    """Altitude in metres of rows `fila` to `fila + filas - 1`, after checking the saved grid."""
    registro = json.loads((carpeta / _REGISTRO).read_text(encoding="utf-8"))
    if registro.get("grilla") != asdict(grilla):
        raise SourceError(f"the slope files in {carpeta} belong to another grid")
    with (carpeta / _ALTITUD).open("rb") as origen:
        origen.seek(fila * grilla.ancho * 2)
        datos = np.fromfile(origen, dtype="<i2", count=filas * grilla.ancho)
    if datos.size != filas * grilla.ancho:
        raise SourceError(f"{carpeta / _ALTITUD} is shorter than its grid")
    return datos.astype(np.int16).reshape(filas, grilla.ancho)
