"""Internal SQLite-backed loader for the bundled refractive-index database.

Owns the read-only connection, decodes binary payloads into the public
dataclasses, and exposes the row-level helpers that `database.py` wraps.
Nothing in this module should be considered public API.

Payload encoding (mirror of `scripts/build_database.py`):
* Tabulated rows: zlib-compressed concatenation of `wavelength_um` blob and
  `values` blob, each `n_points` packed little-endian float32 values.
* Formula rows: raw packed little-endian float32 coefficients in upstream
  order. Layout per formula matches the upstream "Dispersion formulas" doc.
  Coefficient lists shorter than the canonical full length (e.g. a Herzberger
  entry that omits trailing zeros) are zero-padded at decode time.
"""

from __future__ import annotations

import sqlite3
import struct
import zlib
from collections.abc import Iterator
from importlib.resources import files

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


_DB_PATH = str(files("indicio.data").joinpath("refractiveindex.db"))
_conn: sqlite3.Connection | None = None


def _connect() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        uri = f"file:{_DB_PATH}?mode=ro"
        _conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
    return _conn


# --- payload decoding ------------------------------------------------------


def _unpack_floats(payload: bytes) -> list[float]:
    n = len(payload) // 4
    return list(struct.unpack(f"<{n}f", payload))


def _pair_up(values: list[float]) -> tuple[tuple[float, float], ...]:
    # A handful of upstream entries omit a trailing coefficient that the
    # canonical formula expects (implicit zero). Pad rather than reject.
    if len(values) % 2 == 1:
        values = values + [0.0]
    return tuple((values[i], values[i + 1]) for i in range(0, len(values), 2))


def _decode_tabulated(payload: bytes, n_points: int, wr: WavelengthRange) -> Tabulated:
    raw = zlib.decompress(payload)
    half = n_points * 4
    return Tabulated(
        wavelength_um=raw[:half],
        values=raw[half:],
        length=n_points,
        wavelength_range=wr,
    )


def _decode_formula(model_kind: str, payload: bytes, wr: WavelengthRange) -> NModel:
    coeffs = _unpack_floats(payload)
    c1 = coeffs[0] if coeffs else 0.0
    rest = coeffs[1:]

    if model_kind == "formula_1":
        return Sellmeier(c1=c1, coefficients=_pair_up(rest), wavelength_range=wr)
    if model_kind == "formula_2":
        return Sellmeier2(c1=c1, coefficients=_pair_up(rest), wavelength_range=wr)
    if model_kind == "formula_3":
        return Polynomial(c1=c1, terms=_pair_up(rest), wavelength_range=wr)
    if model_kind == "formula_4":
        # Layout: c1, rational_term_1 (B, eB, C, eC), rational_term_2, then
        # polynomial pairs. Rational slots with B=0 are unused upstream.
        rationals_raw = rest[:8] + [0.0] * (8 - len(rest[:8]))
        rationals: list[tuple[float, float, float, float]] = []
        for i in (0, 4):
            B, eB, C, eC = rationals_raw[i : i + 4]
            if B != 0.0:
                rationals.append((B, eB, C, eC))
        return RefractiveIndexInfoFormula4(
            c1=c1,
            rational_terms=tuple(rationals),
            polynomial_terms=_pair_up(rest[8:]),
            wavelength_range=wr,
        )
    if model_kind == "formula_5":
        return Cauchy(c1=c1, terms=_pair_up(rest), wavelength_range=wr)
    if model_kind == "formula_6":
        return Gases(c1=c1, coefficients=_pair_up(rest), wavelength_range=wr)
    if model_kind == "formula_7":
        c = (coeffs + [0.0] * 6)[:6]
        return Herzberger(
            c1=c[0],
            c2=c[1],
            c3=c[2],
            c4=c[3],
            c5=c[4],
            c6=c[5],
            wavelength_range=wr,
        )
    if model_kind == "formula_8":
        c = (coeffs + [0.0] * 4)[:4]
        return Retro(c1=c[0], c2=c[1], c3=c[2], c4=c[3], wavelength_range=wr)
    if model_kind == "formula_9":
        c = (coeffs + [0.0] * 6)[:6]
        return Exotic(
            c1=c[0],
            c2=c[1],
            c3=c[2],
            c4=c[3],
            c5=c[4],
            c6=c[5],
            wavelength_range=wr,
        )
    raise ValueError(f"unknown model_kind: {model_kind!r}")


def _decode_model(
    model_kind: str,
    payload: bytes,
    n_points: int | None,
    wr: WavelengthRange,
) -> NModel | KModel:
    if model_kind == "tabulated":
        assert n_points is not None
        return _decode_tabulated(payload, n_points, wr)
    return _decode_formula(model_kind, payload, wr)


# --- row → MaterialEntry ---------------------------------------------------


def _build_entry(
    shelf: str,
    book: str,
    page: str,
    name: str | None,
    references: str | None,
    comments: str | None,
    model_rows: Iterator[tuple[str, str, bytes, int | None, float, float]],
) -> MaterialEntry:
    n_model: NModel | None = None
    k_model: KModel | None = None
    for quantity, kind, payload, npts, wmn, wmx in model_rows:
        wr: WavelengthRange = (float(wmn), float(wmx))
        if quantity == "n":
            decoded_n: NModel | KModel = _decode_model(kind, payload, npts, wr)
            assert isinstance(
                decoded_n,
                (
                    Tabulated,
                    Sellmeier,
                    Sellmeier2,
                    Polynomial,
                    RefractiveIndexInfoFormula4,
                    Cauchy,
                    Gases,
                    Herzberger,
                    Retro,
                    Exotic,
                ),
            )
            n_model = decoded_n
        elif quantity == "k":
            decoded_k: NModel | KModel = _decode_model(kind, payload, npts, wr)
            assert isinstance(decoded_k, Tabulated)
            k_model = decoded_k
    return MaterialEntry(
        shelf=shelf,
        book=book,
        page=page,
        name=name,
        references=references,
        comments=comments,
        n=n_model,
        k=k_model,
    )


def fetch_material(shelf: str, book: str, page: str) -> MaterialEntry | None:
    conn = _connect()
    head = conn.execute(
        """
        SELECT m.name, r.text, m.comments
          FROM materials m
          LEFT JOIN refs r ON m.ref_id = r.ref_id
         WHERE m.shelf=? AND m.book=? AND m.page=?
        """,
        (shelf, book, page),
    ).fetchone()
    if head is None:
        return None
    name, references, comments = head
    rows = conn.execute(
        """
        SELECT quantity, model_kind, payload, n_points, wl_min, wl_max
          FROM models
         WHERE shelf=? AND book=? AND page=?
        """,
        (shelf, book, page),
    )
    return _build_entry(shelf, book, page, name, references, comments, rows)


def iter_pks() -> Iterator[tuple[str, str, str]]:
    conn = _connect()
    yield from conn.execute(
        "SELECT shelf, book, page FROM materials ORDER BY shelf, book, page"
    )


def iter_materials() -> Iterator[MaterialEntry]:
    """Stream every material entry. Issues one query per material rather than
    materializing the full payload set up front."""
    for shelf, book, page in list(iter_pks()):
        entry = fetch_material(shelf, book, page)
        if entry is not None:
            yield entry


def list_shelves() -> tuple[str, ...]:
    conn = _connect()
    return tuple(
        r[0]
        for r in conn.execute("SELECT DISTINCT shelf FROM materials ORDER BY shelf")
    )


def list_books(shelf: str) -> tuple[str, ...]:
    conn = _connect()
    return tuple(
        r[0]
        for r in conn.execute(
            "SELECT DISTINCT book FROM materials WHERE shelf=? ORDER BY book",
            (shelf,),
        )
    )


def list_pages(shelf: str, book: str) -> tuple[str, ...]:
    conn = _connect()
    return tuple(
        r[0]
        for r in conn.execute(
            "SELECT page FROM materials WHERE shelf=? AND book=? ORDER BY page",
            (shelf, book),
        )
    )


def search_substring(query: str) -> tuple[tuple[str, str, str], ...]:
    conn = _connect()
    pattern = f"%{query}%"
    return tuple(
        (r[0], r[1], r[2])
        for r in conn.execute(
            """
            SELECT shelf, book, page FROM materials
             WHERE book LIKE ? OR page LIKE ?
             ORDER BY shelf, book, page
            """,
            (pattern, pattern),
        )
    )


def database_version() -> str:
    conn = _connect()
    row = conn.execute("SELECT value FROM meta WHERE key='upstream_commit'").fetchone()
    return row[0] if row is not None else ""
