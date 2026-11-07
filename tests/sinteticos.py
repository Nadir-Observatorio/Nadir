"""Synthetic Earth Engine exports: tiled GeoTIFFs inside a zip, as Google Drive hands them back."""

import zipfile
from pathlib import Path

import numpy as np
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

PASO = 0.0002694945852358564
OESTE = -72.0
NORTE = -33.0


def tif(
    datos,
    bandas,
    oeste=OESTE,
    norte=NORTE,
    paso=PASO,
    crs="EPSG:4326",
    sin_dato=255,
    tipo="uint8",
):
    """Bytes of one GeoTIFF with these bands (array of shape bands x rows x columns)."""
    datos = np.asarray(datos)
    with MemoryFile() as memoria:
        with memoria.open(
            driver="GTiff",
            width=datos.shape[2],
            height=datos.shape[1],
            count=datos.shape[0],
            dtype=tipo,
            crs=crs,
            transform=from_origin(oeste, norte, paso, paso),
            nodata=sin_dato,
        ) as destino:
            destino.write(datos.astype(tipo))
            for i, nombre in enumerate(bandas, start=1):
                destino.set_band_description(i, nombre)
        return memoria.read()


def exportar(ruta: Path, prefijo, imagen, bandas, lado, carpeta="exportacion_ee", **opciones):
    """Write `imagen` (bands x rows x columns) into a zip as Earth Engine tiles of side `lado`."""
    imagen = np.asarray(imagen)
    paso = opciones.pop("paso", PASO)
    oeste = opciones.pop("oeste", OESTE)
    norte = opciones.pop("norte", NORTE)
    modo = "a" if ruta.exists() else "w"
    with zipfile.ZipFile(ruta, modo) as archivo:
        for fila in range(0, imagen.shape[1], lado):
            for columna in range(0, imagen.shape[2], lado):
                pedazo = imagen[:, fila : fila + lado, columna : columna + lado]
                contenido = tif(
                    pedazo,
                    bandas,
                    oeste=oeste + columna * paso,
                    norte=norte - fila * paso,
                    paso=paso,
                    **opciones,
                )
                archivo.writestr(f"{carpeta}/{prefijo}-{fila:010d}-{columna:010d}.tif", contenido)
    return ruta
