"""Pixel-to-cell map of an export, built once and reused by every band, year and product.

Assigning a pixel to the H3 cell that holds its centre is the slow step of any aggregation, and
on one export it gives the same answer for every band. So it is done once per resolution and
saved next to the derived data, in its own folder:

- `celdas.npy`: the cells present, as sorted 64-bit H3 indices;
- `indices.u32`: for each pixel, its position in that list (raw little-endian uint32, row by
  row), or `SIN_CELDA` outside the area;
- `mapa.json`: the grid and resolution it belongs to and the content hashes of both files.

Two ways to reach a coarser resolution, and they differ: H3 children do not tile their parent
exactly, so on the MapBiomas grid about 7 % of pixels have a resolution-7 centre cell that is not
the parent of their resolution-8 cell. `tabular` with the map's own resolution assigns by centre;
with a coarser one it adds up child cells, so tables nest exactly across scales.
"""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import h3.api.numpy_int as h3n
import numpy as np
import numpy.typing as npt

from nadir import exportacion
from nadir.errors import GridError, LayerError, ProvenanceError
from nadir.grilla import comprobar_resolucion

SIN_CELDA = np.uint32(0xFFFFFFFF)
"""Index of a pixel outside the area."""

_CELDAS = "celdas.npy"
_INDICES = "indices.u32"
_MAPA = "mapa.json"
_BLOQUE = 1 << 20

Indices = npt.NDArray[np.uint32]


def _sha256(ruta: Path) -> str:
    resumen = hashlib.sha256()
    with ruta.open("rb") as origen:
        while bloque := origen.read(_BLOQUE):
            resumen.update(bloque)
    return resumen.hexdigest()


def celdas_de_franja(
    grilla: exportacion.Grilla, fila: int, dentro: npt.NDArray[np.bool_], resolucion: int
) -> npt.NDArray[np.int64]:
    """Cell holding the centre of every pixel marked `dentro`, in row-major order."""
    filas, columnas = np.nonzero(dentro)
    latitudes = grilla.latitudes(fila, dentro.shape[0])[filas]
    longitudes = grilla.longitudes(0, dentro.shape[1])[columnas]
    convertir = h3n.latlng_to_cell
    return np.fromiter(
        (
            convertir(a, b, resolucion)
            for a, b in zip(latitudes.tolist(), longitudes.tolist(), strict=True)
        ),
        dtype=np.int64,
        count=filas.size,
    )


class MapaCeldas:
    """A saved pixel-to-cell map: the cell list, with pixel indices read from disk on demand."""

    def __init__(
        self,
        carpeta: Path,
        grilla: exportacion.Grilla,
        resolucion: int,
        celdas: npt.NDArray[np.int64],
    ) -> None:
        self.carpeta = carpeta
        self.grilla = grilla
        self.resolucion = resolucion
        self.celdas = celdas

    def indices(self, fila: int, filas: int) -> Indices:
        """Cell positions of rows `fila` to `fila + filas - 1` (a copy, safe to keep)."""
        if fila < 0 or filas < 1 or fila + filas > self.grilla.alto:
            raise GridError(f"rows {fila} to {fila + filas - 1} are outside the map")
        # Read by offset and close at once: a file left mapped stays locked on Windows.
        with (self.carpeta / _INDICES).open("rb") as origen:
            origen.seek(fila * self.grilla.ancho * 4)
            datos = np.fromfile(origen, dtype="<u4", count=filas * self.grilla.ancho)
        if datos.size != filas * self.grilla.ancho:
            raise ProvenanceError(f"{self.carpeta / _INDICES} is shorter than its grid")
        return datos.astype(np.uint32).reshape(filas, self.grilla.ancho)

    def padres(self, resolucion: int) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        """Cells at this or a coarser resolution, and the position of each map cell's parent."""
        if resolucion > self.resolucion:
            raise GridError(f"resolution {resolucion} is finer than the map's ({self.resolucion})")
        if resolucion < 0:
            raise GridError(f"H3 resolutions start at 0, not {resolucion}")
        if resolucion == self.resolucion:
            return self.celdas, np.arange(self.celdas.size, dtype=np.int64)
        padres = np.fromiter(
            (h3n.cell_to_parent(int(c), resolucion) for c in self.celdas.tolist()),
            dtype=np.int64,
            count=self.celdas.size,
        )
        unicas, posicion = np.unique(padres, return_inverse=True)
        return unicas, posicion.astype(np.int64)


def construir(
    mosaico: exportacion.Mosaico, banda: str, resolucion: int, carpeta: Path, alto: int = 512
) -> MapaCeldas:
    """Build the map of an export at one resolution and save it in `carpeta`. Never overwrites.

    A pixel is inside when `banda` is not the export's no-data value; the band only marks the
    area, its values are not used.
    """
    comprobar_resolucion(resolucion)
    if (carpeta / _MAPA).exists() or (carpeta / _INDICES).exists():
        raise LayerError(f"refusing to overwrite the cell map in {carpeta}")
    carpeta.mkdir(parents=True, exist_ok=True)
    grilla = mosaico.grilla
    vistas: dict[int, int] = {}
    parcial = carpeta / f"{_INDICES}.part"
    with parcial.open("wb") as salida:
        for fila, datos in mosaico.franjas(banda, alto):
            dentro = datos != mosaico.sin_dato
            indices = np.full(datos.shape, SIN_CELDA, dtype="<u4")
            if np.any(dentro):
                celdas = celdas_de_franja(grilla, fila, dentro, resolucion)
                unicas, inversa = np.unique(celdas, return_inverse=True)
                posiciones = np.fromiter(
                    (vistas.setdefault(c, len(vistas)) for c in unicas.tolist()),
                    dtype=np.int64,
                    count=unicas.size,
                )
                indices[dentro] = posiciones[inversa].astype("<u4")
            salida.write(indices.tobytes())
    if len(vistas) >= int(SIN_CELDA):
        raise GridError("too many cells for a 32-bit index")
    por_orden = np.fromiter(vistas, dtype=np.int64, count=len(vistas))
    orden = np.argsort(por_orden, kind="stable")
    celdas = por_orden[orden]
    nueva_posicion = np.empty(len(vistas), dtype=np.uint32)
    nueva_posicion[orden] = np.arange(len(vistas), dtype=np.uint32)
    _renumerar(parcial, nueva_posicion, grilla, alto)
    parcial.replace(carpeta / _INDICES)
    np.save(carpeta / _CELDAS, celdas, allow_pickle=False)
    registro = {
        "grilla": asdict(grilla),
        "resolucion": resolucion,
        "banda_de_area": banda,
        "celdas": len(celdas),
        "sha256": {
            _CELDAS: _sha256(carpeta / _CELDAS),
            _INDICES: _sha256(carpeta / _INDICES),
        },
    }
    texto = json.dumps(registro, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    (carpeta / _MAPA).write_bytes(texto.encode("utf-8"))
    return MapaCeldas(carpeta, grilla, resolucion, celdas)


def _renumerar(
    ruta: Path, nueva: npt.NDArray[np.uint32], grilla: exportacion.Grilla, alto: int
) -> None:
    """Rewrite indices in place so they point into the sorted cell list."""
    datos = np.memmap(ruta, dtype="<u4", mode="r+", shape=(grilla.alto, grilla.ancho))
    for fila in range(0, grilla.alto, alto):
        bloque = np.array(datos[fila : fila + alto])
        dentro = bloque != SIN_CELDA
        bloque[dentro] = nueva[bloque[dentro]]
        datos[fila : fila + alto] = bloque
    datos.flush()
    del datos  # drop the only reference so the mapping closes before the file is renamed


def abrir(carpeta: Path, grilla: exportacion.Grilla, comprobar: bool = True) -> MapaCeldas:
    """Open a saved map, refusing it if it belongs to another grid or its files changed."""
    try:
        registro = json.loads((carpeta / _MAPA).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ProvenanceError(f"no cell map in {carpeta}") from None
    if registro.get("grilla") != asdict(grilla):
        raise ProvenanceError(f"the cell map in {carpeta} belongs to another grid")
    if comprobar:
        for nombre, esperada in registro["sha256"].items():
            if _sha256(carpeta / nombre) != esperada:
                raise ProvenanceError(f"{carpeta / nombre}: sha256 differs from {_MAPA}")
    celdas = np.load(carpeta / _CELDAS, allow_pickle=False)
    if celdas.dtype != np.int64 or celdas.size != registro["celdas"]:
        raise ProvenanceError(f"{carpeta / _CELDAS} does not match {_MAPA}")
    return MapaCeldas(carpeta, grilla, int(registro["resolucion"]), celdas)


def tabular(
    mapa: MapaCeldas,
    mosaico: exportacion.Mosaico,
    banda: str,
    resolucion: int | None = None,
    alto: int = 1024,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """Area in m² of every pixel value (0 to 255) in every cell of one band.

    Returns the cells and a table of shape (cells, 256). Without `resolucion`, or with the map's,
    pixels go to the cell that holds their centre. A coarser `resolucion` adds up the map's cells
    into their parents (any resolution from 0, for blocks). Pixels outside the map are left out;
    every value is kept, including the export's no-data value, so callers decide what counts.
    """
    if mosaico.grilla != mapa.grilla:
        raise GridError("the export and the cell map have different grids")
    resolucion = mapa.resolucion if resolucion is None else resolucion
    total = np.zeros((mapa.celdas.size, 256), dtype=np.float64)
    plano = total.reshape(-1)
    for fila, datos in mosaico.franjas(banda, alto):
        indices = mapa.indices(fila, datos.shape[0])
        dentro = indices != SIN_CELDA
        if not np.any(dentro):
            continue
        area = np.broadcast_to(
            mapa.grilla.area_filas_m2(fila, datos.shape[0])[:, None], datos.shape
        )
        clave = indices[dentro].astype(np.int64) * 256 + datos[dentro]
        plano += np.bincount(clave, weights=area[dentro], minlength=plano.size)
    celdas, posicion = mapa.padres(resolucion)
    if resolucion == mapa.resolucion:
        return celdas, total
    agregada = np.zeros((celdas.size, 256), dtype=np.float64)
    np.add.at(agregada, posicion, total)
    return celdas, agregada
