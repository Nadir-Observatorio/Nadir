from pathlib import Path

import pytest

from nadir import grilla, procedencia, tabla
from nadir.errors import LayerError, ProvenanceError
from nadir.nucleo import PROCEDENCIA, Capa, nombre_tabla
from nadir.registro import Variable

VARIABLE = Variable(
    id="prueba.fuego.anual",
    proveedor="Prueba",
    producto="Fuego sintético",
    version="1",
    licencia="CC-BY-4.0",
    atribucion="Prueba",
    unidad="fracción",
    tiempo=(2020, 2021),
    agregacion="fraccion",
    sin_dato="excluido",
    url="https://ejemplo.org/",
)

CELDA = grilla.celda(-33.03, -71.55, 7)


class CapaSintetica(Capa):
    """A layer whose 'download' writes a small file and whose tables are fixed."""

    def __init__(self, crudos: Path, salida: Path, anio: int = 2020, mezclar: bool = False) -> None:
        super().__init__(VARIABLE, crudos, salida)
        self.anio = anio
        self.mezclar = mezclar

    def descargar(self) -> list[procedencia.Original]:
        ruta = self.crudos / "fuente.bin"
        ruta.parent.mkdir(parents=True, exist_ok=True)
        ruta.write_bytes(b"pixeles")
        return [procedencia.registrar("https://ejemplo.org/f", self.crudos, ruta, "2026-11-04")]

    def derivar(self, originales):
        filas = [tabla.Fila(CELDA, self.anio, 0.5, 1.0)]
        if self.mezclar:
            filas.append(tabla.Fila(grilla.celda(-38.74, -72.60, 7), self.anio + 1, 0.1, 1.0))
        return {
            (7, self.anio): filas,
            (6, self.anio): [tabla.Fila(grilla.padre(CELDA, 6), self.anio, 0.5, 0.1)],
        }


def test_table_names():
    assert nombre_tabla(7, 2020) == "r7_2020.parquet"


def test_build_writes_tables_and_provenance_then_verifies(tmp_path):
    capa = CapaSintetica(tmp_path / "crudos", tmp_path / "salida")
    huellas = capa.construir()
    assert sorted(huellas) == ["r6_2020.parquet", "r7_2020.parquet"]
    variable, originales, derivados = procedencia.leer(tmp_path / "salida" / PROCEDENCIA)
    assert variable == VARIABLE.id
    assert derivados == huellas
    assert originales[0].ruta == "fuente.bin"
    capa.verificar()


def test_build_refuses_to_run_twice(tmp_path):
    capa = CapaSintetica(tmp_path / "crudos", tmp_path / "salida")
    capa.construir()
    with pytest.raises(LayerError, match="already built"):
        capa.construir()


def test_year_outside_registry_period_is_refused(tmp_path):
    capa = CapaSintetica(tmp_path / "crudos", tmp_path / "salida", anio=2019)
    with pytest.raises(LayerError, match="outside the registry period"):
        capa.construir()


def test_table_mixing_years_is_refused(tmp_path):
    capa = CapaSintetica(tmp_path / "crudos", tmp_path / "salida", mezclar=True)
    with pytest.raises(LayerError, match="mixes years"):
        capa.construir()


def test_verify_detects_changed_original(tmp_path):
    capa = CapaSintetica(tmp_path / "crudos", tmp_path / "salida")
    capa.construir()
    (tmp_path / "crudos" / "fuente.bin").write_bytes(b"pixelez")
    with pytest.raises(ProvenanceError, match="sha256"):
        capa.verificar()


def test_verify_detects_extra_or_changed_table(tmp_path):
    capa = CapaSintetica(tmp_path / "crudos", tmp_path / "salida")
    capa.construir()
    extra = tmp_path / "salida" / "r5_2020.parquet"
    tabla.escribir(extra, [tabla.Fila(grilla.padre(CELDA, 5), 2020, 0.5, 0.01)])
    with pytest.raises(ProvenanceError, match="differ from the record"):
        capa.verificar()
    extra.unlink()
    r7 = tmp_path / "salida" / "r7_2020.parquet"
    r7.unlink()
    tabla.escribir(r7, [tabla.Fila(CELDA, 2020, 0.6, 1.0)])
    with pytest.raises(ProvenanceError, match="content hash"):
        capa.verificar()


class CapaVacia(CapaSintetica):
    def descargar(self):
        return []


def test_empty_download_is_refused(tmp_path):
    with pytest.raises(LayerError, match="no files"):
        CapaVacia(tmp_path / "crudos", tmp_path / "salida").construir()
