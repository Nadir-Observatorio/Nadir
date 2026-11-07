"""The common cell table: one per variable and resolution (docs/design.md, section 3).

Columns: `celda` (H3 id), `tiempo` (year), `valor`, `fraccion_valida`. Rows are sorted by cell
and year. The content hash is computed from the values, not from the Parquet bytes, so it does
not depend on the pyarrow version, compression or operating system.
"""

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import h3
import pyarrow as pa
import pyarrow.parquet as pq

from nadir.errors import TableError

ESQUEMA = pa.schema(
    [
        ("celda", pa.string()),
        ("tiempo", pa.int32()),
        ("valor", pa.float64()),
        ("fraccion_valida", pa.float64()),
    ]
)


@dataclass(frozen=True, order=True)
class Fila:
    """One row of a cell table."""

    celda: str
    tiempo: int
    valor: float
    fraccion_valida: float


def _validar(filas: list[Fila]) -> list[Fila]:
    vistas: set[tuple[str, int]] = set()
    for fila in filas:
        if not h3.is_valid_cell(fila.celda):
            raise TableError(f"not a valid H3 cell: {fila.celda!r}")
        if not math.isfinite(fila.valor):
            raise TableError(f"{fila.celda} {fila.tiempo}: value must be finite")
        if not 0.0 <= fila.fraccion_valida <= 1.0:
            raise TableError(f"{fila.celda} {fila.tiempo}: fraccion_valida out of [0, 1]")
        clave = (fila.celda, fila.tiempo)
        if clave in vistas:
            raise TableError(f"duplicated row {clave}")
        vistas.add(clave)
    return sorted(filas)


def huella(filas: list[Fila]) -> str:
    """Content hash of a table: sha256 over the sorted rows with exact float representation."""
    resumen = hashlib.sha256()
    for fila in _validar(filas):
        linea = (
            f"{fila.celda}\t{fila.tiempo}\t{float(fila.valor).hex()}\t"
            f"{float(fila.fraccion_valida).hex()}\n"
        )
        resumen.update(linea.encode("ascii"))
    return resumen.hexdigest()


def escribir(ruta: Path, filas: list[Fila]) -> str:
    """Write a table as Parquet and return its content hash. Refuses to overwrite."""
    if ruta.exists():
        raise TableError(f"refusing to overwrite {ruta}")
    ordenadas = _validar(filas)
    tabla = pa.table(
        {
            "celda": [f.celda for f in ordenadas],
            "tiempo": [f.tiempo for f in ordenadas],
            "valor": [f.valor for f in ordenadas],
            "fraccion_valida": [f.fraccion_valida for f in ordenadas],
        },
        schema=ESQUEMA,
    )
    ruta.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(tabla, ruta, compression="zstd")
    return huella(ordenadas)


def leer(ruta: Path) -> list[Fila]:
    """Read a table written by `escribir`, checking its schema."""
    try:
        tabla = pq.read_table(ruta)
    except FileNotFoundError:
        raise TableError(f"table not found: {ruta}") from None
    if not tabla.schema.equals(ESQUEMA):
        raise TableError(f"{ruta.name}: schema differs from the common cell table")
    columnas = tabla.to_pydict()
    filas = [
        Fila(str(c), int(t), float(v), float(f))
        for c, t, v, f in zip(
            columnas["celda"],
            columnas["tiempo"],
            columnas["valor"],
            columnas["fraccion_valida"],
            strict=True,
        )
    ]
    return _validar(filas)
