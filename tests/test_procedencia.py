import hashlib

import pytest

from nadir import procedencia
from nadir.errors import ProvenanceError


def _original(tmp_path, contenido=b"datos"):
    ruta = tmp_path / "fuente" / "archivo.tif"
    ruta.parent.mkdir(parents=True)
    ruta.write_bytes(contenido)
    return procedencia.registrar("https://ejemplo.org/a.tif", tmp_path, ruta, "2026-11-04")


def test_record_measures_size_and_hash(tmp_path):
    original = _original(tmp_path)
    assert original.ruta == "fuente/archivo.tif"
    assert original.bytes == 5
    assert original.sha256 == hashlib.sha256(b"datos").hexdigest()


def test_large_file_hash_matches_hashlib(tmp_path):
    contenido = bytes(range(256)) * 9000
    original = _original(tmp_path, contenido)
    assert original.sha256 == hashlib.sha256(contenido).hexdigest()


def test_file_outside_root_is_refused(tmp_path):
    otra = tmp_path.parent / f"{tmp_path.name}-fuera.tif"
    otra.write_bytes(b"x")
    with pytest.raises(ProvenanceError, match="not inside"):
        procedencia.registrar("https://ejemplo.org/", tmp_path, otra, "2026-11-04")


def test_check_passes_and_detects_changes(tmp_path):
    original = _original(tmp_path)
    procedencia.comprobar(original, tmp_path)
    (tmp_path / original.ruta).write_bytes(b"datoz")
    with pytest.raises(ProvenanceError, match="sha256"):
        procedencia.comprobar(original, tmp_path)
    (tmp_path / original.ruta).write_bytes(b"otro largo")
    with pytest.raises(ProvenanceError, match="size changed"):
        procedencia.comprobar(original, tmp_path)
    (tmp_path / original.ruta).unlink()
    with pytest.raises(ProvenanceError, match="missing"):
        procedencia.comprobar(original, tmp_path)


def test_write_read_roundtrip_and_bytes_are_deterministic(tmp_path):
    original = _original(tmp_path)
    ruta = tmp_path / "salida" / "procedencia.json"
    procedencia.escribir(
        ruta, "a.b.c", [original], {"r7_2020.parquet": "ff", "r6_2020.parquet": "aa"}
    )
    contenido = ruta.read_bytes()
    assert b"\r\n" not in contenido
    assert contenido.endswith(b"\n")
    variable, originales, derivados = procedencia.leer(ruta)
    assert variable == "a.b.c"
    assert originales == [original]
    assert list(derivados) == ["r6_2020.parquet", "r7_2020.parquet"]
    otra = tmp_path / "otra" / "procedencia.json"
    procedencia.escribir(
        otra, "a.b.c", [original], {"r6_2020.parquet": "aa", "r7_2020.parquet": "ff"}
    )
    assert otra.read_bytes() == contenido


def test_write_refuses_to_overwrite(tmp_path):
    ruta = tmp_path / "procedencia.json"
    ruta.write_text("{}", encoding="utf-8")
    with pytest.raises(ProvenanceError, match="overwrite"):
        procedencia.escribir(ruta, "a.b.c", [], {})


def test_read_reports_bad_files(tmp_path):
    with pytest.raises(ProvenanceError, match="not found"):
        procedencia.leer(tmp_path / "no.json")
    roto = tmp_path / "roto.json"
    roto.write_text("{", encoding="utf-8")
    with pytest.raises(ProvenanceError, match="not valid JSON"):
        procedencia.leer(roto)
    incompleto = tmp_path / "incompleto.json"
    incompleto.write_text('{"variable": "a"}', encoding="utf-8")
    with pytest.raises(ProvenanceError, match="wrong structure"):
        procedencia.leer(incompleto)
