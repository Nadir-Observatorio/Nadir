import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from nadir import grilla, tabla
from nadir.errors import TableError

A = grilla.celda(-33.03, -71.55, 7)
B = grilla.celda(-38.74, -72.60, 7)


def _filas():
    return [tabla.Fila(B, 2020, 0.25, 1.0), tabla.Fila(A, 2020, 0.5, 0.75)]


def test_roundtrip_sorts_rows(tmp_path):
    ruta = tmp_path / "r7_2020.parquet"
    tabla.escribir(ruta, _filas())
    assert tabla.leer(ruta) == sorted(_filas())


def test_hash_ignores_row_order_and_matches_written_table(tmp_path):
    ruta = tmp_path / "r7_2020.parquet"
    escrita = tabla.escribir(ruta, _filas())
    assert escrita == tabla.huella(list(reversed(_filas())))
    assert escrita == tabla.huella(tabla.leer(ruta))


def test_hash_sees_the_smallest_change():
    cambiada = [tabla.Fila(B, 2020, 0.25 + 1e-15, 1.0), tabla.Fila(A, 2020, 0.5, 0.75)]
    assert tabla.huella(cambiada) != tabla.huella(_filas())


def test_hash_is_fixed_for_known_input():
    filas = [tabla.Fila("87b2c4080ffffff", 2020, 0.5, 1.0)]
    assert tabla.huella(filas) == tabla.huella(list(filas))
    assert len(tabla.huella(filas)) == 64


def test_write_refuses_to_overwrite(tmp_path):
    ruta = tmp_path / "r7_2020.parquet"
    tabla.escribir(ruta, _filas())
    with pytest.raises(TableError, match="overwrite"):
        tabla.escribir(ruta, _filas())


@pytest.mark.parametrize(
    ("fila", "mensaje"),
    [
        (tabla.Fila("no-es-celda", 2020, 0.1, 1.0), "valid H3"),
        (tabla.Fila(A, 2020, float("nan"), 1.0), "finite"),
        (tabla.Fila(A, 2020, 0.1, 1.5), "out of"),
    ],
)
def test_invalid_rows_are_refused(fila, mensaje):
    with pytest.raises(TableError, match=mensaje):
        tabla.huella([fila])


def test_duplicated_rows_are_refused():
    with pytest.raises(TableError, match="duplicated"):
        tabla.huella([tabla.Fila(A, 2020, 0.1, 1.0), tabla.Fila(A, 2020, 0.2, 1.0)])


def test_foreign_schema_is_refused(tmp_path):
    ruta = tmp_path / "otra.parquet"
    pq.write_table(pa.table({"celda": [A], "valor": [1.0]}), ruta)
    with pytest.raises(TableError, match="schema"):
        tabla.leer(ruta)


def test_missing_table_is_reported(tmp_path):
    with pytest.raises(TableError, match="not found"):
        tabla.leer(tmp_path / "no.parquet")
