"""Public lookup and browsing API for the bundled refractive-index database.

Thin wrappers over `_loader`. All wavelengths are micrometers, matching the
upstream convention.
"""

from __future__ import annotations

from collections.abc import Iterator

from indicio._loader import (
    database_version as _db_version,
    fetch_material as _fetch_material,
    iter_materials as _iter_materials,
    list_books as _list_books,
    list_pages as _list_pages,
    list_shelves as _list_shelves,
    search_substring as _search_substring,
)
from indicio.models import MaterialEntry


def get_material(shelf: str, book: str, page: str) -> MaterialEntry:
    """Return the material entry for `(shelf, book, page)`.

    Raises `KeyError` if the triple is unknown.
    """
    entry = _fetch_material(shelf, book, page)
    if entry is None:
        raise KeyError((shelf, book, page))
    return entry


def has_material(shelf: str, book: str, page: str) -> bool:
    return _fetch_material(shelf, book, page) is not None


def shelves() -> tuple[str, ...]:
    return _list_shelves()


def books(shelf: str) -> tuple[str, ...]:
    return _list_books(shelf)


def pages(shelf: str, book: str) -> tuple[str, ...]:
    return _list_pages(shelf, book)


def iter_materials() -> Iterator[MaterialEntry]:
    """Stream every material entry. Useful for building search indices or
    exporting to other formats."""
    return _iter_materials()


def search(query: str) -> tuple[tuple[str, str, str], ...]:
    """Substring search over book and page names.

    Returns `(shelf, book, page)` triples sorted lexicographically.
    """
    return _search_substring(query)


def database_version() -> str:
    """Upstream commit hash of the bundled YAML database snapshot."""
    return _db_version()
