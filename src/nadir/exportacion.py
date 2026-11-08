"""Google Earth Engine exports kept in the raw-data bank, read in place from their zip files.

Some sources (MapBiomas Chile among them) serve their rasters only through Earth Engine. Nadir
exports them to Google Drive with a script, and the zip that Drive hands back goes into the bank
untouched, with its `procedencia.toml` entry. Nothing is unzipped to disk: GeoTIFFs are read
through GDAL's zip support.

An export of one image arrives as tiles named `<prefix>-<row>-<column>.tif`, where row and column
are the pixel offsets of the tile, ten digits each. Before any pixel is used, `abrir` checks that
the tiles are one image: same reference system, cell size, band names, data type and no-data
value; each tile's origin where its name says; no tile twice; and together a full rectangle with
no gap or overlap.
"""

import re
import zipfile
from collections.abc import Iterator
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Self

import numpy as np
import numpy.typing as npt
import rasterio
from rasterio.errors import RasterioIOError
from rasterio.io import DatasetReader
from rasterio.windows import Window

from nadir.area import area_filas_m2
from nadir.errors import GridError, SourceError

SRC = "EPSG:4326"
"""Reference system of every export Nadir reads: the native grid of its sources."""

_TESELA = re.compile(r"^(?:.*/)?(?P<prefijo>[^/]+)-(?P<fila>\d{10})-(?P<columna>\d{10})\.tif$")
_TOLERANCIA = 1e-6
"""Largest accepted misplacement of a tile origin, as a fraction of one cell."""

Pixeles = npt.NDArray[np.uint8]


@dataclass(frozen=True)
class Grilla:
    """A north-up grid of square cells in degrees: top-left corner, cell size and shape."""

    oeste: float
    norte: float
    paso: float
    ancho: int
    alto: int

    def latitudes(self, fila: int, filas: int) -> npt.NDArray[np.float64]:
        """Latitude of the centre of each row in `fila` to `fila + filas - 1`."""
        indices = np.arange(fila, fila + filas, dtype=np.float64)
        resultado: npt.NDArray[np.float64] = self.norte - (indices + 0.5) * self.paso
        return resultado

    def longitudes(self, columna: int, columnas: int) -> npt.NDArray[np.float64]:
        """Longitude of the centre of each column in `columna` to `columna + columnas - 1`."""
        indices = np.arange(columna, columna + columnas, dtype=np.float64)
        resultado: npt.NDArray[np.float64] = self.oeste + (indices + 0.5) * self.paso
        return resultado

    def area_filas_m2(self, fila: int, filas: int) -> npt.NDArray[np.float64]:
        """Area in m² of one pixel of each row in `fila` to `fila + filas - 1`."""
        return area_filas_m2(self.norte - fila * self.paso, self.paso, filas)


def desfase(base: Grilla, otra: Grilla) -> tuple[int, int]:
    """Whole (column, row) offset of `otra`'s corner on `base`'s grid.

    Raises unless both grids share the cell size and `otra` starts on a cell edge of `base`, so a
    pixel of one is exactly a pixel of the other and nothing needs resampling.
    """
    if abs(base.paso - otra.paso) > base.paso * 1e-9:
        raise GridError(f"cell sizes differ: {base.paso} and {otra.paso}")
    columnas = (otra.oeste - base.oeste) / base.paso
    filas = (base.norte - otra.norte) / base.paso
    columna, fila = round(columnas), round(filas)
    if abs(columnas - columna) > _TOLERANCIA or abs(filas - fila) > _TOLERANCIA:
        raise GridError(
            f"grids are not aligned: offset of {columnas:.6f} columns and {filas:.6f} rows"
        )
    return columna, fila


def misma_grilla(a: Grilla, b: Grilla) -> bool:
    """Whether two grids put every pixel in the same place.

    Exports of different products can carry cell sizes that differ in the last bit of their
    floating-point value. They are the same grid when shapes match, corners align and the size
    difference, added up across the whole grid, stays below a millionth of a cell.
    """
    if (a.ancho, a.alto) != (b.ancho, b.alto):
        return False
    if abs(a.paso - b.paso) * max(a.ancho, a.alto) > a.paso * _TOLERANCIA:
        return False
    try:
        return desfase(a, b) == (0, 0)
    except GridError:
        return False


@dataclass(frozen=True)
class Tesela:
    """One tile of an export: where it is stored and where it sits in the full image."""

    zip: Path
    miembro: str
    fila: int
    columna: int
    alto: int
    ancho: int

    @property
    def ruta(self) -> str:
        """GDAL path of the tile inside its zip."""
        return f"/vsizip/{self.zip.resolve().as_posix()}/{self.miembro}"


class Mosaico:
    """An export opened for reading: its grid, band names and tiles.

    Use it as a context manager; the tiles stay open until it closes.
    """

    def __init__(
        self,
        prefijo: str,
        grilla: Grilla,
        bandas: tuple[str, ...],
        sin_dato: int,
        teselas: tuple[Tesela, ...],
    ) -> None:
        self.prefijo = prefijo
        self.grilla = grilla
        self.bandas = bandas
        self.sin_dato = sin_dato
        self.teselas = teselas
        self._pila = ExitStack()
        self._abiertas: dict[Tesela, DatasetReader] = {}

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        tipo: type[BaseException] | None,
        error: BaseException | None,
        traza: TracebackType | None,
    ) -> None:
        self.cerrar()

    def cerrar(self) -> None:
        """Close every open tile."""
        self._pila.close()
        self._abiertas.clear()

    def _abierta(self, tesela: Tesela) -> DatasetReader:
        if tesela not in self._abiertas:
            self._abiertas[tesela] = self._pila.enter_context(rasterio.open(tesela.ruta))
        return self._abiertas[tesela]

    def indice(self, banda: str) -> int:
        """1-based index of a band, by name."""
        try:
            return self.bandas.index(banda) + 1
        except ValueError:
            raise SourceError(f"{self.prefijo}: no band {banda!r}") from None

    def leer(self, banda: str, fila: int, filas: int) -> Pixeles:
        """Rows `fila` to `fila + filas - 1` of one band, full width."""
        if fila < 0 or filas < 1 or fila + filas > self.grilla.alto:
            raise GridError(f"rows {fila} to {fila + filas - 1} are outside the image")
        indice = self.indice(banda)
        salida = np.full((filas, self.grilla.ancho), self.sin_dato, dtype=np.uint8)
        for tesela in self.teselas:
            inicio = max(fila, tesela.fila)
            fin = min(fila + filas, tesela.fila + tesela.alto)
            if inicio >= fin:
                continue
            ventana = Window(0, inicio - tesela.fila, tesela.ancho, fin - inicio)
            datos = self._abierta(tesela).read(indice, window=ventana)
            salida[inicio - fila : fin - fila, tesela.columna : tesela.columna + tesela.ancho] = (
                datos
            )
        return salida

    def franjas(self, banda: str, alto: int) -> Iterator[tuple[int, Pixeles]]:
        """Walk one band from north to south in full-width strips of at most `alto` rows."""
        if alto < 1:
            raise GridError(f"strip height must be positive, not {alto}")
        for fila in range(0, self.grilla.alto, alto):
            yield fila, self.leer(banda, fila, min(alto, self.grilla.alto - fila))


def _miembros(zips: list[Path], prefijo: str) -> list[tuple[Path, str, int, int]]:
    encontrados: list[tuple[Path, str, int, int]] = []
    for ruta in zips:
        try:
            with zipfile.ZipFile(ruta) as archivo:
                nombres = archivo.namelist()
        except (OSError, zipfile.BadZipFile) as error:
            raise SourceError(f"{ruta}: not a readable zip ({error})") from None
        for nombre in nombres:
            coincide = _TESELA.match(nombre)
            if coincide and coincide["prefijo"] == prefijo:
                encontrados.append((ruta, nombre, int(coincide["fila"]), int(coincide["columna"])))
    return encontrados


def _continua(inicios: dict[int, int], eje: str) -> int:
    """Check that tiles along one axis follow each other with no gap; return the total size."""
    posicion = 0
    for inicio in sorted(inicios):
        if inicio != posicion:
            raise GridError(f"tiles leave a gap or overlap along the {eje} at {posicion}")
        posicion += inicios[inicio]
    return posicion


def abrir(zips: list[Path], prefijo: str) -> Mosaico:
    """Open the tiles of export `prefijo` in these zips, after checking they form one image."""
    miembros = _miembros(zips, prefijo)
    if not miembros:
        raise SourceError(f"no tiles named {prefijo}-<row>-<column>.tif in {len(zips)} zip(s)")
    posiciones = [(fila, columna) for _, _, fila, columna in miembros]
    if len(set(posiciones)) != len(posiciones):
        raise SourceError(f"{prefijo}: the same tile appears more than once")

    teselas: list[Tesela] = []
    referencia: tuple[str, tuple[str, ...], str, float | None, float] | None = None
    esquina: tuple[float, float] | None = None
    for ruta, nombre, fila, columna in sorted(miembros, key=lambda m: (m[2], m[3])):
        tesela_ruta = Tesela(ruta, nombre, fila, columna, 0, 0).ruta
        try:
            with rasterio.open(tesela_ruta) as datos:
                transformada = datos.transform
                crs = datos.crs.to_string() if datos.crs else ""
                bandas = tuple(str(d) for d in datos.descriptions)
                tipos = set(datos.dtypes)
                sin_dato = datos.nodata
                alto, ancho = datos.height, datos.width
        except RasterioIOError as error:
            raise SourceError(f"{nombre}: cannot be read ({error})") from None
        if crs != SRC:
            raise SourceError(f"{nombre}: reference system {crs!r}, expected {SRC}")
        if tipos != {"uint8"}:
            raise SourceError(f"{nombre}: data type {sorted(tipos)}, expected uint8")
        if transformada.b != 0 or transformada.d != 0 or transformada.e != -transformada.a:
            raise SourceError(f"{nombre}: the grid is rotated or its cells are not square")
        paso = float(transformada.a)
        actual = (crs, bandas, "uint8", sin_dato, paso)
        if referencia is None:
            referencia = actual
            esquina = (
                float(transformada.c) - columna * paso,
                float(transformada.f) + fila * paso,
            )
        elif actual[:4] != referencia[:4] or abs(paso - referencia[4]) > referencia[4] * 1e-9:
            raise SourceError(f"{nombre}: bands, no-data value or cell size differ from the rest")
        if esquina is None:
            raise SourceError(f"{prefijo}: no reference tile")
        paso_ref = referencia[4]
        error_x = (float(transformada.c) - (esquina[0] + columna * paso_ref)) / paso_ref
        error_y = (float(transformada.f) - (esquina[1] - fila * paso_ref)) / paso_ref
        if abs(error_x) > _TOLERANCIA or abs(error_y) > _TOLERANCIA:
            raise GridError(f"{nombre}: its origin is not where its name places it")
        teselas.append(Tesela(ruta, nombre, fila, columna, alto, ancho))

    if referencia is None or esquina is None:
        raise SourceError(f"{prefijo}: no tiles")
    if referencia[3] is None:
        raise SourceError(f"{prefijo}: the tiles declare no no-data value")
    altos: dict[int, int] = {}
    anchos: dict[int, int] = {}
    for tesela in teselas:
        if altos.setdefault(tesela.fila, tesela.alto) != tesela.alto:
            raise GridError(f"{prefijo}: tiles of row {tesela.fila} have different heights")
        if anchos.setdefault(tesela.columna, tesela.ancho) != tesela.ancho:
            raise GridError(f"{prefijo}: tiles of column {tesela.columna} have different widths")
    alto_total = _continua(altos, "rows")
    ancho_total = _continua(anchos, "columns")
    if len(teselas) != len(altos) * len(anchos):
        raise SourceError(
            f"{prefijo}: {len(teselas)} tiles, expected {len(altos) * len(anchos)} "
            f"({len(altos)} rows x {len(anchos)} columns)"
        )
    grilla = Grilla(esquina[0], esquina[1], referencia[4], ancho_total, alto_total)
    return Mosaico(prefijo, grilla, referencia[1], int(referencia[3]), tuple(teselas))
