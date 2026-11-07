"""The variable registry: every variable of the observatory, with its source, licence and rules.

The registry lives in `registro/variables.toml`. A variable enters only with a verified licence
(docs/design.md, principle 2); its aggregation rule says how its pixels become H3 cells.
"""

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nadir.errors import RegistryError
from nadir.licencias import licencia

REGLAS = ("fraccion", "suma", "media", "moda", "conteo")
"""Closed set of aggregation rules (docs/design.md, section 3)."""

CAMPOS = (
    "id",
    "proveedor",
    "producto",
    "version",
    "licencia",
    "atribucion",
    "unidad",
    "tiempo",
    "agregacion",
    "sin_dato",
    "url",
)
"""Mandatory fields of every variable, in the order they are documented."""

_ID = re.compile(r"^[a-z0-9]+(\.[a-z0-9_]+)+$")


@dataclass(frozen=True)
class Variable:
    """One variable of the observatory, exactly as declared in the registry."""

    id: str
    proveedor: str
    producto: str
    version: str
    licencia: str
    atribucion: str
    unidad: str
    tiempo: tuple[int, int]
    agregacion: str
    sin_dato: str
    url: str


def _texto(entrada: dict[str, Any], campo: str, ident: str) -> str:
    valor = entrada[campo]
    if not isinstance(valor, str) or not valor.strip():
        raise RegistryError(f"{ident}: field {campo!r} must be a non-empty string")
    return valor


def _tiempo(entrada: dict[str, Any], ident: str) -> tuple[int, int]:
    valor = entrada["tiempo"]
    if (
        not isinstance(valor, list)
        or len(valor) != 2
        or not all(isinstance(v, int) and not isinstance(v, bool) for v in valor)
    ):
        raise RegistryError(f"{ident}: 'tiempo' must be [first_year, last_year]")
    inicio, fin = valor
    if inicio > fin:
        raise RegistryError(f"{ident}: 'tiempo' starts after it ends ({inicio} > {fin})")
    return (inicio, fin)


def _variable(entrada: dict[str, Any], posicion: int) -> Variable:
    ident = str(entrada.get("id", f"variable #{posicion}"))
    faltan = [c for c in CAMPOS if c not in entrada]
    if faltan:
        raise RegistryError(f"{ident}: missing fields {', '.join(faltan)}")
    sobran = sorted(set(entrada) - set(CAMPOS))
    if sobran:
        raise RegistryError(f"{ident}: unknown fields {', '.join(sobran)}")
    if not isinstance(entrada["id"], str) or not _ID.match(entrada["id"]):
        raise RegistryError(f"{ident}: id must look like 'provider.product.detail' in lowercase")
    agregacion = _texto(entrada, "agregacion", ident)
    if agregacion not in REGLAS:
        raise RegistryError(f"{ident}: aggregation {agregacion!r} is not one of {REGLAS}")
    url = _texto(entrada, "url", ident)
    if not url.startswith("https://"):
        raise RegistryError(f"{ident}: url must be https")
    licencia(_texto(entrada, "licencia", ident))
    return Variable(
        id=entrada["id"],
        proveedor=_texto(entrada, "proveedor", ident),
        producto=_texto(entrada, "producto", ident),
        version=_texto(entrada, "version", ident),
        licencia=entrada["licencia"],
        atribucion=_texto(entrada, "atribucion", ident),
        unidad=_texto(entrada, "unidad", ident),
        tiempo=_tiempo(entrada, ident),
        agregacion=agregacion,
        sin_dato=_texto(entrada, "sin_dato", ident),
        url=url,
    )


def cargar(ruta: Path) -> dict[str, Variable]:
    """Read and validate the registry. Returns the variables by id, in file order."""
    try:
        datos = tomllib.loads(ruta.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise RegistryError(f"registry not found: {ruta}") from None
    except tomllib.TOMLDecodeError as error:
        raise RegistryError(f"registry is not valid TOML ({ruta.name}): {error}") from None
    entradas = datos.get("variable")
    if not isinstance(entradas, list) or not entradas:
        raise RegistryError(f"{ruta.name}: no [[variable]] entries")
    variables: dict[str, Variable] = {}
    for posicion, entrada in enumerate(entradas, start=1):
        if not isinstance(entrada, dict):
            raise RegistryError(f"{ruta.name}: entry #{posicion} is not a table")
        variable = _variable(entrada, posicion)
        if variable.id in variables:
            raise RegistryError(f"{ruta.name}: duplicated id {variable.id!r}")
        variables[variable.id] = variable
    return variables
