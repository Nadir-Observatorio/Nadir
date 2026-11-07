"""The layer engine: download, derive, check (docs/design.md, section 5).

A layer turns one registry variable into cell tables:

1. `descargar` fetches the originals into the raw-data folder and returns their records;
2. `derivar` builds one table per (resolution, year) from the originals;
3. `construir` runs both, writes the tables and `procedencia.json`, and never overwrites;
4. `verificar` checks the originals and the tables on disk against `procedencia.json`.

A concrete layer implements only `descargar` and `derivar`; everything that guarantees the chain
of custody lives here, once.
"""

from abc import ABC, abstractmethod
from pathlib import Path

from nadir import procedencia, tabla
from nadir.errors import LayerError, ProvenanceError
from nadir.grilla import comprobar_resolucion
from nadir.registro import Variable

PROCEDENCIA = "procedencia.json"


def nombre_tabla(resolucion: int, anio: int) -> str:
    """File name of the table for one resolution and year."""
    comprobar_resolucion(resolucion)
    return f"r{resolucion}_{anio}.parquet"


class Capa(ABC):
    """One variable of the registry, turned into cell tables with full provenance."""

    def __init__(self, variable: Variable, crudos: Path, salida: Path) -> None:
        self.variable = variable
        self.crudos = crudos
        self.salida = salida

    @abstractmethod
    def descargar(self) -> list[procedencia.Original]:
        """Fetch the originals under `self.crudos` and return one record per file."""

    @abstractmethod
    def derivar(
        self, originales: list[procedencia.Original]
    ) -> dict[tuple[int, int], list[tabla.Fila]]:
        """Build the tables from the originals, keyed by (resolution, year)."""

    def construir(self) -> dict[str, str]:
        """Download, derive and write everything. Returns content hashes by table name."""
        ruta_procedencia = self.salida / PROCEDENCIA
        if ruta_procedencia.exists():
            raise LayerError(f"{self.variable.id}: already built ({ruta_procedencia})")
        originales = self.descargar()
        if not originales:
            raise LayerError(f"{self.variable.id}: download returned no files")
        for original in originales:
            procedencia.comprobar(original, self.crudos)
        tablas = self.derivar(originales)
        if not tablas:
            raise LayerError(f"{self.variable.id}: derive returned no tables")
        inicio, fin = self.variable.tiempo
        huellas: dict[str, str] = {}
        for (resolucion, anio), filas in sorted(tablas.items()):
            if not inicio <= anio <= fin:
                raise LayerError(
                    f"{self.variable.id}: year {anio} outside the registry period {inicio}-{fin}"
                )
            nombre = nombre_tabla(resolucion, anio)
            if filas and any(f.tiempo != anio for f in filas):
                raise LayerError(f"{self.variable.id}: table {nombre} mixes years")
            huellas[nombre] = tabla.escribir(self.salida / nombre, filas)
        procedencia.escribir(ruta_procedencia, self.variable.id, originales, huellas)
        return huellas

    def verificar(self) -> None:
        """Raise unless originals and tables on disk match `procedencia.json`."""
        variable, originales, huellas = procedencia.leer(self.salida / PROCEDENCIA)
        if variable != self.variable.id:
            raise ProvenanceError(f"provenance belongs to {variable!r}, not {self.variable.id!r}")
        for original in originales:
            procedencia.comprobar(original, self.crudos)
        en_disco = sorted(p.name for p in self.salida.glob("r*_*.parquet"))
        if en_disco != sorted(huellas):
            raise ProvenanceError(
                f"{self.variable.id}: tables on disk {en_disco} differ from the record"
            )
        for nombre, esperada in huellas.items():
            if tabla.huella(tabla.leer(self.salida / nombre)) != esperada:
                raise ProvenanceError(f"{self.variable.id}: {nombre} content hash differs")
