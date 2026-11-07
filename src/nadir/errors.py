"""Errors raised by nadir. Every failure has its own type so callers can tell them apart."""


class NadirError(Exception):
    """Base class for every error raised by nadir."""


class RegistryError(NadirError):
    """The variable registry is missing, malformed or inconsistent."""


class LicenseError(NadirError):
    """A licence is unknown, unverified or cannot be combined."""


class ProvenanceError(NadirError):
    """A source file or a provenance record does not match what was recorded."""


class GridError(NadirError):
    """An H3 resolution, cell or aggregation request is invalid."""


class TableError(NadirError):
    """A cell table has the wrong schema or invalid values."""


class LayerError(NadirError):
    """A layer cannot be built: missing inputs, existing outputs or a failed check."""


class SourceError(NadirError):
    """A download failed or a source does not serve what it should."""
