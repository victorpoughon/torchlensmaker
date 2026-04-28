"""Offline access to the refractiveindex.info database.

The package ships a bundled SQLite snapshot of the upstream YAML database
and exposes every dispersion model as a typed dataclass. It does not
compute n(λ) or k(λ) — that's left to consumers.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as _pkg_version

from indicio.database import (
    books,
    database_version as _database_version,
    get_material,
    has_material,
    iter_materials,
    pages,
    search,
    shelves,
)
from indicio.models import (
    Cauchy,
    Exotic,
    Gases,
    Herzberger,
    KModel,
    MaterialEntry,
    NModel,
    Polynomial,
    RefractiveIndexInfoFormula4,
    Retro,
    Sellmeier,
    Sellmeier2,
    Tabulated,
    WavelengthRange,
)

try:
    __version__: str = _pkg_version("indicio")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__database_version__: str = _database_version()


__all__ = [
    "Cauchy",
    "Exotic",
    "Gases",
    "Herzberger",
    "KModel",
    "MaterialEntry",
    "NModel",
    "Polynomial",
    "RefractiveIndexInfoFormula4",
    "Retro",
    "Sellmeier",
    "Sellmeier2",
    "Tabulated",
    "WavelengthRange",
    "books",
    "get_material",
    "has_material",
    "iter_materials",
    "pages",
    "search",
    "shelves",
    "__version__",
    "__database_version__",
]
