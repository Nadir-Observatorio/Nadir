"""The raw-data bank: one folder per source and dataset, with its own `procedencia.toml`.

Bank rule: an original file is never edited. If a source changes, it is downloaded again under
another name and a new `[[archivo]]` entry is added. `procedencia.toml` starts with a `[fuente]`
block and gets one `[[archivo]]` entry per file; entries are only appended, never rewritten.

Downloads go to a `.part` file first, are checked against the size the server announced, and
are renamed only when complete. Network failures are retried a few times; client errors (4xx)
fail at once.
"""

import hashlib
import time
import tomllib
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from http.client import HTTPResponse
from pathlib import Path
from typing import Any, cast

from nadir.errors import ProvenanceError, SourceError

PROCEDENCIA = "procedencia.toml"
AGENTE = "nadir/0.1"
_BLOQUE = 1 << 20
_REINTENTABLES = {429, 500, 502, 503, 504}


@dataclass(frozen=True)
class Fuente:
    """Header of a bank folder: who publishes the dataset and where it is offered."""

    organismo: str
    conjunto: str
    fuente_pagina: str
    licencia: str
    atribucion: str


@dataclass(frozen=True)
class Archivo:
    """One downloaded file of a bank folder."""

    nombre: str
    periodo: str
    url: str
    url_verificada: bool
    fecha_descarga: str
    bytes: int
    sha256: str


def _toml_texto(valor: str) -> str:
    escapado = valor.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escapado}"'


def _toml_valor(valor: str | int | bool) -> str:
    if isinstance(valor, bool):
        return "true" if valor else "false"
    if isinstance(valor, int):
        return str(valor)
    return _toml_texto(valor)


def _bloque(titulo: str, campos: dict[str, str | int | bool]) -> str:
    lineas = [titulo] + [f"{clave} = {_toml_valor(valor)}" for clave, valor in campos.items()]
    return "\n".join(lineas) + "\n"


def leer(carpeta: Path) -> tuple[Fuente, list[Archivo]]:
    """Read a folder's `procedencia.toml`."""
    ruta = carpeta / PROCEDENCIA
    try:
        datos: dict[str, Any] = tomllib.loads(ruta.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ProvenanceError(f"no {PROCEDENCIA} in {carpeta}") from None
    except tomllib.TOMLDecodeError as error:
        raise ProvenanceError(f"{ruta} is not valid TOML: {error}") from None
    try:
        fuente = Fuente(**datos["fuente"])
        archivos = [Archivo(**a) for a in datos.get("archivo", [])]
    except (KeyError, TypeError) as error:
        raise ProvenanceError(f"{ruta} has the wrong structure: {error}") from None
    return fuente, archivos


def preparar(carpeta: Path, fuente: Fuente) -> None:
    """Create the folder and its `[fuente]` block, or check that the existing one matches."""
    ruta = carpeta / PROCEDENCIA
    if ruta.exists():
        existente, _ = leer(carpeta)
        if existente != fuente:
            raise ProvenanceError(f"{ruta} describes another source: {existente}")
        return
    carpeta.mkdir(parents=True, exist_ok=True)
    encabezado = (
        f"# Bank provenance. Original files are never edited; entries are only appended.\n"
        f"# Written by nadir on {datetime.now().astimezone().date().isoformat()}.\n\n"
    )
    campos: dict[str, str | int | bool] = {
        "organismo": fuente.organismo,
        "conjunto": fuente.conjunto,
        "fuente_pagina": fuente.fuente_pagina,
        "licencia": fuente.licencia,
        "atribucion": fuente.atribucion,
    }
    ruta.write_bytes((encabezado + _bloque("[fuente]", campos)).encode("utf-8"))


def agregar(carpeta: Path, archivo: Archivo) -> None:
    """Append one `[[archivo]]` entry. Refuses a name that is already recorded."""
    _, archivos = leer(carpeta)
    if any(a.nombre == archivo.nombre for a in archivos):
        raise ProvenanceError(f"{archivo.nombre} is already recorded in {carpeta / PROCEDENCIA}")
    campos: dict[str, str | int | bool] = {
        "nombre": archivo.nombre,
        "periodo": archivo.periodo,
        "url": archivo.url,
        "url_verificada": archivo.url_verificada,
        "fecha_descarga": archivo.fecha_descarga,
        "bytes": archivo.bytes,
        "sha256": archivo.sha256,
    }
    with (carpeta / PROCEDENCIA).open("ab") as destino:
        destino.write(("\n" + _bloque("[[archivo]]", campos)).encode("utf-8"))


def comprobar(carpeta: Path) -> list[Archivo]:
    """Raise unless every recorded file is on disk with its recorded size and sha256."""
    _, archivos = leer(carpeta)
    for archivo in archivos:
        ruta = carpeta / archivo.nombre
        if not ruta.is_file():
            raise ProvenanceError(f"recorded file is missing: {ruta}")
        if ruta.stat().st_size != archivo.bytes:
            raise ProvenanceError(f"{ruta}: size differs from the record")
        if _sha256(ruta) != archivo.sha256:
            raise ProvenanceError(f"{ruta}: sha256 differs from the record")
    return archivos


def registrar(
    carpeta: Path,
    nombre: str,
    periodo: str,
    url: str,
    url_verificada: bool,
    fecha_descarga: str | None = None,
) -> Archivo:
    """Record a file that is already in a prepared bank folder (for example, from Google Drive).

    Size and sha256 are measured now. Without `fecha_descarga`, the file's modification time is
    used, as the closest record of when it arrived.
    """
    ruta = carpeta / nombre
    if Path(nombre).name != nombre:
        raise ProvenanceError(f"{nombre!r} must be a file name inside {carpeta}, not a path")
    if not ruta.is_file():
        raise ProvenanceError(f"file not found: {ruta}")
    if fecha_descarga is None:
        momento = datetime.fromtimestamp(ruta.stat().st_mtime).astimezone()
        fecha_descarga = momento.isoformat(timespec="seconds")
    archivo = Archivo(
        nombre=nombre,
        periodo=periodo,
        url=url,
        url_verificada=url_verificada,
        fecha_descarga=fecha_descarga,
        bytes=ruta.stat().st_size,
        sha256=_sha256(ruta),
    )
    agregar(carpeta, archivo)
    return archivo


def zips(carpeta: Path) -> list[Path]:
    """Recorded zip files of a folder, after checking every recorded file against its record."""
    return [carpeta / a.nombre for a in comprobar(carpeta) if a.nombre.lower().endswith(".zip")]


def _sha256(ruta: Path) -> str:
    resumen = hashlib.sha256()
    with ruta.open("rb") as origen:
        while bloque := origen.read(_BLOQUE):
            resumen.update(bloque)
    return resumen.hexdigest()


def _abrir(solicitud: urllib.request.Request, espera: float, intentos: int) -> HTTPResponse:
    for intento in range(1, intentos + 1):
        try:
            respuesta = urllib.request.urlopen(solicitud, timeout=60)  # noqa: S310 (https only)
            return cast(HTTPResponse, respuesta)
        except urllib.error.HTTPError as error:
            if error.code not in _REINTENTABLES or intento == intentos:
                raise SourceError(f"{solicitud.full_url}: HTTP {error.code}") from None
        except (urllib.error.URLError, TimeoutError) as error:
            if intento == intentos:
                raise SourceError(f"{solicitud.full_url}: {error}") from None
        time.sleep(espera * intento)
    raise SourceError(f"{solicitud.full_url}: no attempts left")


def _https(url: str) -> None:
    if not url.startswith("https://"):
        raise SourceError(f"only https downloads are allowed: {url}")


def tamano_remoto(url: str, espera: float = 2.0, intentos: int = 3) -> int:
    """Size announced by the server for a file, without downloading it (HEAD request)."""
    _https(url)
    solicitud = urllib.request.Request(  # noqa: S310 (https enforced above)
        url, method="HEAD", headers={"User-Agent": AGENTE}
    )
    with _abrir(solicitud, espera, intentos) as respuesta:
        largo = respuesta.headers.get("Content-Length")
    if largo is None:
        raise SourceError(f"{url}: the server did not announce a size")
    return int(largo)


def estado_remoto(url: str) -> tuple[int, int | None]:
    """HTTP status and announced size of a file (HEAD request), without retries or raising.

    For diagnosis only: a 403 from a public cloud bucket usually means the object does not exist
    there, because the bucket does not allow listing.
    """
    _https(url)
    solicitud = urllib.request.Request(  # noqa: S310 (https enforced above)
        url, method="HEAD", headers={"User-Agent": AGENTE}
    )
    try:
        with urllib.request.urlopen(solicitud, timeout=60) as respuesta:  # noqa: S310
            largo = respuesta.headers.get("Content-Length")
            return int(respuesta.status), int(largo) if largo is not None else None
    except urllib.error.HTTPError as error:
        return int(error.code), None
    except (urllib.error.URLError, TimeoutError) as error:
        raise SourceError(f"{url}: {error}") from None


def descargar(
    url: str,
    carpeta: Path,
    nombre: str,
    periodo: str,
    url_verificada: bool,
    espera: float = 2.0,
    intentos: int = 3,
) -> Archivo:
    """Download one file into a prepared bank folder and record it.

    Refuses to overwrite. The file is written as `<name>.part`, checked against the announced
    size, renamed, hashed and appended to `procedencia.toml`.
    """
    _https(url)
    destino = carpeta / nombre
    parcial = carpeta / f"{nombre}.part"
    if destino.exists():
        raise SourceError(f"refusing to overwrite {destino}")
    leer(carpeta)
    solicitud = urllib.request.Request(url, headers={"User-Agent": AGENTE})  # noqa: S310
    resumen = hashlib.sha256()
    escritos = 0
    with _abrir(solicitud, espera, intentos) as respuesta:
        anunciado = respuesta.headers.get("Content-Length")
        try:
            with parcial.open("wb") as salida:
                while bloque := respuesta.read(_BLOQUE):
                    salida.write(bloque)
                    resumen.update(bloque)
                    escritos += len(bloque)
        except (OSError, urllib.error.URLError, TimeoutError) as error:
            raise SourceError(
                f"{url}: download cut off ({error}); partial file {parcial}"
            ) from None
    if anunciado is not None and int(anunciado) != escritos:
        raise SourceError(
            f"{url}: got {escritos} bytes, server announced {anunciado}; see {parcial}"
        )
    parcial.replace(destino)
    archivo = Archivo(
        nombre=nombre,
        periodo=periodo,
        url=url,
        url_verificada=url_verificada,
        fecha_descarga=datetime.now().astimezone().isoformat(timespec="seconds"),
        bytes=escritos,
        sha256=resumen.hexdigest(),
    )
    agregar(carpeta, archivo)
    return archivo
