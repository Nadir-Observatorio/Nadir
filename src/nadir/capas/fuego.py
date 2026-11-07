"""MapBiomas Fire Chile, collection 1: burned area by year and by month, from the bank.

Exported from Earth Engine with `gee/exportar_mapbiomas_7_regiones.js` in the native grid of the
assets (EPSG:4326, the same grid as land cover collection 2), over the seven regions the product
maps (Valparaíso to La Araucanía), years 2013 to 2025. Both exports come in one zip.

- Annual (`burned_area_<year>`): 1 burned, 0 not burned, 255 outside the seven regions.
- Monthly (`burned_monthly_<year>`): 1 to 12 is the month a pixel burned in that year, 0 not
  burned, 255 outside. One month per pixel and year.
- Fire frequency is derived here from the annual bands, not exported: one source, one record.

A year band is a calendar year: a fire in December belongs to that December's year.

Scars are the connected groups of pixels burned in the same year, eight neighbours. They are
labelled strip by strip and joined across strips, so the whole export never has to sit in memory.
"""

import csv
from collections import Counter
from dataclasses import astuple, dataclass, fields
from pathlib import Path

import numpy as np
import numpy.typing as npt
from scipy import ndimage

from nadir import banco, exportacion
from nadir.area import distancia_km
from nadir.errors import LayerError, SourceError

ACTIVOS = (
    "projects/mapbiomas-chile/assets/FIRE/COLLECTION1/FINAL_PRODUCTS/"
    "mapbiomas_chile_fire_collection1_annual_burned_v1",
    "projects/mapbiomas-chile/assets/FIRE/COLLECTION1/FINAL_PRODUCTS/"
    "mapbiomas_chile_fire_collection1_monthly_burned_v1",
)
PAGINA = "https://mapbiomas-chile.projects.earthengine.app/view/chile-lulc-descargas"
PREFIJO_ANUAL = "mbchile_fuego_c1_anual_2013_2025"
PREFIJO_MENSUAL = "mbchile_fuego_c1_mensual_2013_2025"
ANIOS = tuple(range(2013, 2026))
CARPETA = Path("mapbiomas-chile") / "fuego-c1"
NO_QUEMADO = 0
QUEMADO = 1
FUERA = 255
FRANJA = 1024

FUENTE = banco.Fuente(
    organismo="MapBiomas Chile",
    conjunto="Area Quemada (MapBiomas Fuego Chile), Coleccion 1 (2013-2025), anual y mensual",
    fuente_pagina=PAGINA,
    licencia="MapBiomas-Chile",
    atribucion=(
        "MapBiomas Chile Project - Collection 1 of MapBiomas Fire Chile, consulted on "
        "[date] via the link: https://chile.mapbiomas.org"
    ),
)


def banda_anual(anio: int) -> str:
    """Annual band name of one year."""
    _comprobar_anio(anio)
    return f"burned_area_{anio}"


def banda_mensual(anio: int) -> str:
    """Monthly band name of one year."""
    _comprobar_anio(anio)
    return f"burned_monthly_{anio}"


def _comprobar_anio(anio: int) -> None:
    if anio not in ANIOS:
        raise SourceError(f"collection 1 covers {ANIOS[0]}-{ANIOS[-1]}, not {anio}")


def _abrir(zips: list[Path], prefijo: str, bandas: tuple[str, ...]) -> exportacion.Mosaico:
    mosaico = exportacion.abrir(zips, prefijo)
    if mosaico.bandas != bandas or mosaico.sin_dato != FUERA:
        mosaico.cerrar()
        raise SourceError(
            f"{prefijo}: bands {mosaico.bandas} and no-data {mosaico.sin_dato}, "
            f"expected {bandas} and {FUERA}"
        )
    return mosaico


def abrir(bruto: Path) -> tuple[exportacion.Mosaico, exportacion.Mosaico]:
    """Open the annual and monthly exports, after checking the bank and that they share a grid."""
    zips = banco.zips(bruto / CARPETA)
    anual = _abrir(zips, PREFIJO_ANUAL, tuple(banda_anual(a) for a in ANIOS))
    try:
        mensual = _abrir(zips, PREFIJO_MENSUAL, tuple(banda_mensual(a) for a in ANIOS))
    except SourceError:
        anual.cerrar()
        raise
    if anual.grilla != mensual.grilla:
        anual.cerrar()
        mensual.cerrar()
        raise SourceError("annual and monthly exports have different grids")
    return anual, mensual


def frecuencia(anual: exportacion.Mosaico, fila: int, filas: int) -> exportacion.Pixeles:
    """Number of years each pixel burned in 2013-2025, for rows `fila` to `fila + filas - 1`.

    255 outside the seven regions, as in the exports.
    """
    total = np.zeros((filas, anual.grilla.ancho), dtype=np.uint8)
    fuera: npt.NDArray[np.bool_] | None = None
    for anio in ANIOS:
        datos = anual.leer(banda_anual(anio), fila, filas)
        total += (datos == QUEMADO).astype(np.uint8)
        fuera = datos == FUERA if fuera is None else fuera
    if fuera is not None:
        total[fuera] = FUERA
    return total


@dataclass(frozen=True)
class Informe:
    """Burned hectares by year and by month, and pixel counts, for the exports."""

    grilla: exportacion.Grilla
    hectareas_anio: dict[int, float]
    hectareas_mes: dict[int, dict[int, float]]
    pixeles_anio: dict[int, Counter[int]]


def describir(
    anual: exportacion.Mosaico, mensual: exportacion.Mosaico, anios: tuple[int, ...] = ANIOS
) -> Informe:
    """Describe both exports and refuse them if they disagree.

    Checks, pixel by pixel and year by year: annual values in {0, 1, 255}, monthly in 0 to 12 or
    255; burned in the annual exactly where the monthly has a month; outside in both at once.
    """
    hectareas_anio: dict[int, float] = {}
    hectareas_mes: dict[int, dict[int, float]] = {}
    pixeles_anio: dict[int, Counter[int]] = {}
    for anio in anios:
        quemada = 0.0
        por_mes = np.zeros(256, dtype=np.float64)
        cuenta = np.zeros(256, dtype=np.int64)
        for fila, a in anual.franjas(banda_anual(anio), FRANJA):
            m = mensual.leer(banda_mensual(anio), fila, a.shape[0])
            if np.any((a != NO_QUEMADO) & (a != QUEMADO) & (a != FUERA)):
                raise SourceError(f"{anio}: annual values outside 0, 1 and 255")
            if np.any((m > 12) & (m != FUERA)):
                raise SourceError(f"{anio}: monthly values outside 0 to 12 and 255")
            if np.any((a == QUEMADO) != ((m >= 1) & (m <= 12))):
                raise SourceError(f"{anio}: annual and monthly disagree on burned pixels")
            if np.any((a == FUERA) != (m == FUERA)):
                raise SourceError(f"{anio}: annual and monthly disagree on the seven regions")
            pesos = np.repeat(anual.grilla.area_filas_m2(fila, a.shape[0]), a.shape[1])
            quemada += float(np.sum(pesos[a.ravel() == QUEMADO]))
            por_mes += np.bincount(m.ravel(), weights=pesos, minlength=256)
            cuenta += np.bincount(a.ravel(), minlength=256)
        hectareas_anio[anio] = quemada / 10_000
        hectareas_mes[anio] = {
            mes: float(por_mes[mes]) / 10_000 for mes in range(1, 13) if por_mes[mes] > 0
        }
        pixeles_anio[anio] = Counter({int(c): int(cuenta[c]) for c in np.flatnonzero(cuenta)})
    return Informe(anual.grilla, hectareas_anio, hectareas_mes, pixeles_anio)


_VECINOS = np.ones((3, 3), dtype=bool)


@dataclass(frozen=True)
class Cicatriz:
    """One fire scar: pixels burned in the same year and connected through their 8 neighbours.

    The box is the smallest rectangle of whole pixels that holds the scar; its diagonal runs from
    the south-west to the north-east corner, on the mean sphere.
    """

    anio: int
    numero: int
    pixeles: int
    hectareas: float
    norte: float
    sur: float
    oeste: float
    este: float
    diagonal_km: float


class _Uniones:
    """Union-find over scar labels, with the stats of each label kept at its root."""

    def __init__(self) -> None:
        self.padre: list[int] = []

    def nuevos(self, cantidad: int) -> int:
        base = len(self.padre)
        self.padre.extend(range(base, base + cantidad))
        return base

    def raiz(self, etiqueta: int) -> int:
        while self.padre[etiqueta] != etiqueta:
            self.padre[etiqueta] = self.padre[self.padre[etiqueta]]
            etiqueta = self.padre[etiqueta]
        return etiqueta

    def unir(self, a: int, b: int) -> None:
        ra, rb = self.raiz(a), self.raiz(b)
        if ra != rb:
            self.padre[max(ra, rb)] = min(ra, rb)


def cicatrices(anual: exportacion.Mosaico, anio: int, alto: int = FRANJA) -> list[Cicatriz]:
    """Every scar of one year, numbered from north-west to south-east."""
    if alto < 1:
        raise LayerError(f"strip height must be positive, not {alto}")
    grilla = anual.grilla
    uniones = _Uniones()
    pixeles: list[npt.NDArray[np.int64]] = []
    areas: list[npt.NDArray[np.float64]] = []
    fila_min: list[npt.NDArray[np.int64]] = []
    fila_max: list[npt.NDArray[np.int64]] = []
    col_min: list[npt.NDArray[np.int64]] = []
    col_max: list[npt.NDArray[np.int64]] = []
    borde: npt.NDArray[np.int64] | None = None
    for fila, datos in anual.franjas(banda_anual(anio), alto):
        etiquetas, cantidad = ndimage.label(datos == QUEMADO, structure=_VECINOS)
        etiquetas = etiquetas.astype(np.int64)
        base = uniones.nuevos(cantidad)
        globales = np.where(etiquetas > 0, etiquetas - 1 + base, -1)
        if cantidad:
            filas, columnas = np.nonzero(etiquetas)
            locales = etiquetas[filas, columnas] - 1
            orden = np.argsort(locales, kind="stable")
            locales, filas, columnas = locales[orden], filas[orden], columnas[orden]
            inicios = np.flatnonzero(np.r_[True, np.diff(locales) != 0])
            area_fila = grilla.area_filas_m2(fila, datos.shape[0])
            pixeles.append(np.bincount(locales, minlength=cantidad).astype(np.int64))
            areas.append(np.bincount(locales, weights=area_fila[filas], minlength=cantidad))
            fila_min.append(np.minimum.reduceat(filas, inicios) + fila)
            fila_max.append(np.maximum.reduceat(filas, inicios) + fila)
            col_min.append(np.minimum.reduceat(columnas, inicios))
            col_max.append(np.maximum.reduceat(columnas, inicios))
        if borde is not None:
            primera = globales[0]
            for paso in (-1, 0, 1):
                a = borde[max(0, -paso) : len(borde) - max(0, paso)]
                b = primera[max(0, paso) : len(primera) - max(0, -paso)]
                juntas = (a >= 0) & (b >= 0)
                for x, y in zip(a[juntas].tolist(), b[juntas].tolist(), strict=True):
                    uniones.unir(x, y)
        borde = globales[-1]
    if not pixeles:
        return []
    raices = np.asarray([uniones.raiz(e) for e in range(len(uniones.padre))], dtype=np.int64)
    unicas, grupo = np.unique(raices, return_inverse=True)
    n = len(unicas)
    total_pixeles = np.bincount(grupo, weights=np.concatenate(pixeles), minlength=n)
    total_areas = np.bincount(grupo, weights=np.concatenate(areas), minlength=n)
    fmin = np.full(n, np.iinfo(np.int64).max)
    fmax = np.full(n, -1)
    cmin = np.full(n, np.iinfo(np.int64).max)
    cmax = np.full(n, -1)
    np.minimum.at(fmin, grupo, np.concatenate(fila_min))
    np.maximum.at(fmax, grupo, np.concatenate(fila_max))
    np.minimum.at(cmin, grupo, np.concatenate(col_min))
    np.maximum.at(cmax, grupo, np.concatenate(col_max))
    norte = grilla.norte - fmin * grilla.paso
    sur = grilla.norte - (fmax + 1) * grilla.paso
    oeste = grilla.oeste + cmin * grilla.paso
    este = grilla.oeste + (cmax + 1) * grilla.paso
    diagonal = distancia_km(sur, oeste, norte, este)
    orden = np.lexsort((cmin, fmin))
    return [
        Cicatriz(
            anio=anio,
            numero=numero,
            pixeles=int(total_pixeles[i]),
            hectareas=float(total_areas[i]) / 10_000,
            norte=float(norte[i]),
            sur=float(sur[i]),
            oeste=float(oeste[i]),
            este=float(este[i]),
            diagonal_km=float(diagonal[i]),
        )
        for numero, i in enumerate(orden.tolist(), start=1)
    ]


def escribir_cicatrices(ruta: Path, lista: list[Cicatriz]) -> None:
    """Write scars as CSV (UTF-8, LF, fixed decimals) so the file is the same on any system.

    Refuses to overwrite.
    """
    if ruta.exists():
        raise LayerError(f"refusing to overwrite {ruta}")
    ruta.parent.mkdir(parents=True, exist_ok=True)
    nombres = [f.name for f in fields(Cicatriz)]
    with ruta.open("w", encoding="utf-8", newline="") as destino:
        escritor = csv.writer(destino, lineterminator="\n")
        escritor.writerow(nombres)
        for c in lista:
            fila = list(astuple(c))
            escritor.writerow(
                [
                    c.anio,
                    c.numero,
                    c.pixeles,
                    f"{c.hectareas:.4f}",
                    *(f"{v:.9f}" for v in fila[4:8]),
                    f"{c.diagonal_km:.4f}",
                ]
            )
