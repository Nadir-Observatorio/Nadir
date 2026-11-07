"""Known licences and how a product inherits them from its inputs (docs/design.md, section 4).

A product that combines several variables carries the most restrictive licence among its inputs.
Only licences listed here can enter the observatory; anything else is refused, because a layer
without a verified licence is never published (docs/design.md, principle 2).

MapBiomas Chile states its terms in two ways: the platform notes say CC BY, and its FAQ says the
data are free "for non-commercial use or for public interest purposes". Until the network clarifies
it in writing, its data carry their own entry, read conservatively: attribution plus a possible
non-commercial condition. A possible non-commercial condition cannot be combined with a
share-alike licence (CC BY-SA requires allowing commercial reuse), so that combination is refused.
"""

from dataclasses import dataclass

from nadir.errors import LicenseError


@dataclass(frozen=True)
class Licencia:
    """A licence the observatory accepts.

    `rango` orders licences from least to most restrictive. `comparte_igual` marks share-alike
    licences, whose derivatives must keep the same licence. `no_comercial` marks terms that may
    forbid commercial reuse.
    """

    codigo: str
    nombre: str
    url: str
    rango: int
    comparte_igual: bool
    no_comercial: bool = False


LICENCIAS: dict[str, Licencia] = {
    "CC0-1.0": Licencia(
        "CC0-1.0",
        "Creative Commons Zero 1.0",
        "https://creativecommons.org/publicdomain/zero/1.0/",
        0,
        False,
    ),
    "PDDL-1.0": Licencia(
        "PDDL-1.0",
        "Open Data Commons Public Domain Dedication and License 1.0",
        "https://opendatacommons.org/licenses/pddl/1-0/",
        0,
        False,
    ),
    "CC-BY-4.0": Licencia(
        "CC-BY-4.0",
        "Creative Commons Attribution 4.0",
        "https://creativecommons.org/licenses/by/4.0/",
        1,
        False,
    ),
    "CC-BY-SA-4.0": Licencia(
        "CC-BY-SA-4.0",
        "Creative Commons Attribution-ShareAlike 4.0",
        "https://creativecommons.org/licenses/by-sa/4.0/",
        2,
        True,
    ),
    "Copernicus-DEM-GLO-30": Licencia(
        "Copernicus-DEM-GLO-30",
        "Licence for the use of the Copernicus WorldDEM-30 (free, worldwide, with notice)",
        "https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Data/DEM/resources/license/License-COPDEM-30.pdf",
        1,
        False,
    ),
    "MapBiomas-Chile": Licencia(
        "MapBiomas-Chile",
        "MapBiomas Chile terms: CC BY (platform notes); FAQ: non-commercial or public interest",
        "https://chile.mapbiomas.org/terminos-de-uso/",
        2,
        False,
        True,
    ),
}


def licencia(codigo: str) -> Licencia:
    """Return the licence with this code, or raise if the observatory does not accept it."""
    try:
        return LICENCIAS[codigo]
    except KeyError:
        aceptadas = ", ".join(sorted(LICENCIAS))
        raise LicenseError(f"licence {codigo!r} is not accepted; accepted: {aceptadas}") from None


def heredada(codigos: list[str]) -> Licencia:
    """Licence of a product built from inputs with these licences: the most restrictive wins.

    Two different share-alike licences of the same rank cannot be combined, because each one
    requires its own terms for the derivative.
    """
    if not codigos:
        raise LicenseError("a product needs at least one input licence")
    entradas = [licencia(c) for c in codigos]
    if any(e.no_comercial for e in entradas) and any(e.comparte_igual for e in entradas):
        conflicto = sorted({e.codigo for e in entradas if e.no_comercial or e.comparte_igual})
        raise LicenseError(
            "a possibly non-commercial licence cannot be combined with a share-alike one: "
            + ", ".join(conflicto)
        )
    maximo = max(e.rango for e in entradas)
    candidatas = sorted({e.codigo for e in entradas if e.rango == maximo})
    if len(candidatas) > 1 and any(LICENCIAS[c].comparte_igual for c in candidatas):
        raise LicenseError(f"share-alike licences cannot be combined: {', '.join(candidatas)}")
    return LICENCIAS[candidatas[0]]
