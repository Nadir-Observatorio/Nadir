"""Provenance: what was downloaded, from where, when, and what it hashes to.

Every original file is recorded with its URL, size, sha256 and download date. The record is
written as JSON with sorted keys, UTF-8 and LF line endings, so it is byte-identical on Windows
and Linux.
"""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from nadir.errors import ProvenanceError

_BLOQUE = 1 << 20


def sha256_archivo(ruta: Path) -> str:
    """sha256 of a file, read in 1 MiB blocks so large rasters do not fill memory."""
    resumen = hashlib.sha256()
    try:
        with ruta.open("rb") as archivo:
            while bloque := archivo.read(_BLOQUE):
                resumen.update(bloque)
    except FileNotFoundError:
        raise ProvenanceError(f"file not found: {ruta}") from None
    return resumen.hexdigest()


@dataclass(frozen=True)
class Original:
    """One downloaded source file. `ruta` is relative to the raw-data folder."""

    url: str
    ruta: str
    bytes: int
    sha256: str
    fecha: str


def registrar(url: str, raiz: Path, ruta: Path, fecha: str) -> Original:
    """Record a file that already sits under `raiz`, measuring its size and hash now."""
    try:
        relativa = ruta.resolve().relative_to(raiz.resolve())
    except ValueError:
        raise ProvenanceError(f"{ruta.name} is not inside {raiz}") from None
    if not ruta.is_file():
        raise ProvenanceError(f"file not found: {ruta}")
    return Original(
        url=url,
        ruta=relativa.as_posix(),
        bytes=ruta.stat().st_size,
        sha256=sha256_archivo(ruta),
        fecha=fecha,
    )


def comprobar(original: Original, raiz: Path) -> None:
    """Raise if the file on disk no longer matches its record (size first, then hash)."""
    ruta = raiz / original.ruta
    if not ruta.is_file():
        raise ProvenanceError(f"recorded file is missing: {original.ruta}")
    tamano = ruta.stat().st_size
    if tamano != original.bytes:
        raise ProvenanceError(
            f"{original.ruta}: size changed ({tamano} bytes, recorded {original.bytes})"
        )
    if sha256_archivo(ruta) != original.sha256:
        raise ProvenanceError(f"{original.ruta}: sha256 differs from the record")


def a_bytes(registro: dict[str, Any]) -> bytes:
    """Deterministic JSON bytes: sorted keys, two-space indent, UTF-8, LF, final newline."""
    texto = json.dumps(registro, ensure_ascii=False, indent=2, sort_keys=True)
    return (texto + "\n").encode("utf-8")


def escribir(
    ruta: Path, variable: str, originales: list[Original], derivados: dict[str, str]
) -> None:
    """Write `procedencia.json` for one layer. Refuses to overwrite an existing record."""
    if ruta.exists():
        raise ProvenanceError(f"refusing to overwrite {ruta}")
    registro = {
        "variable": variable,
        "originales": [asdict(o) for o in sorted(originales, key=lambda o: o.ruta)],
        "derivados": dict(sorted(derivados.items())),
    }
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_bytes(a_bytes(registro))


def leer(ruta: Path) -> tuple[str, list[Original], dict[str, str]]:
    """Read a `procedencia.json` written by `escribir`."""
    try:
        datos = json.loads(ruta.read_bytes().decode("utf-8"))
    except FileNotFoundError:
        raise ProvenanceError(f"provenance not found: {ruta}") from None
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProvenanceError(f"{ruta.name} is not valid JSON: {error}") from None
    try:
        originales = [Original(**o) for o in datos["originales"]]
        derivados = {str(k): str(v) for k, v in datos["derivados"].items()}
        return str(datos["variable"]), originales, derivados
    except (KeyError, TypeError) as error:
        raise ProvenanceError(f"{ruta.name} has the wrong structure: {error}") from None
