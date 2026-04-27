"""Smoke + integrity tests against the bundled SQLite database.

These tests run on the artifact produced by `scripts/build_database.py`. They
check schema integrity, expected counts, and that a few well-known materials
round-trip their numeric content faithfully.
"""

from __future__ import annotations

import math
import struct
import zlib

import pytest


# --- payload decoders ---------------------------------------------------


def decode_formula(payload: bytes) -> tuple[float, ...]:
    """Decode a formula payload (raw little-endian float32 array)."""
    n = len(payload) // 4
    return struct.unpack(f"<{n}f", payload)


def decode_tabulated(
    payload: bytes, n_points: int
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Decode a tabulated payload: zlib-inflated, split into wavelengths and values."""
    raw = zlib.decompress(payload)
    expected = n_points * 2 * 4
    assert len(raw) == expected, f"expected {expected} bytes, got {len(raw)}"
    half = n_points
    wls = struct.unpack(f"<{half}f", raw[: half * 4])
    vals = struct.unpack(f"<{half}f", raw[half * 4 :])
    return wls, vals


# --- expected counts ----------------------------------------------------


def test_materials_count(conn):
    (n,) = conn.execute("SELECT COUNT(*) FROM materials").fetchone()
    assert n >= 3500


def test_models_count(conn):
    (n,) = conn.execute("SELECT COUNT(*) FROM models").fetchone()
    assert n >= 6000


def test_refs_count(conn):
    (n,) = conn.execute("SELECT COUNT(*) FROM refs").fetchone()
    assert n >= 700


def test_meta_has_upstream_commit(conn):
    (val,) = conn.execute(
        "SELECT value FROM meta WHERE key='upstream_commit'"
    ).fetchone()
    # 40 hex chars or short form — just sanity check it's a hex string.
    assert val and all(c in "0123456789abcdef" for c in val)


# --- schema integrity ---------------------------------------------------


def test_no_orphan_model_rows(conn):
    """Every (shelf, book, page) referenced by `models` exists in `materials`."""
    rows = conn.execute(
        "SELECT COUNT(*) FROM models m"
        " WHERE NOT EXISTS ("
        "   SELECT 1 FROM materials x"
        "   WHERE x.shelf=m.shelf AND x.book=m.book AND x.page=m.page)"
    ).fetchone()
    assert rows[0] == 0


def test_no_orphan_ref_ids(conn):
    """Every non-NULL materials.ref_id resolves to a refs row."""
    (n,) = conn.execute(
        "SELECT COUNT(*) FROM materials m"
        " WHERE m.ref_id IS NOT NULL"
        "   AND NOT EXISTS (SELECT 1 FROM refs r WHERE r.ref_id = m.ref_id)"
    ).fetchone()
    assert n == 0


def test_known_model_kinds(conn):
    """Every model_kind matches the expected set."""
    kinds = {row[0] for row in conn.execute("SELECT DISTINCT model_kind FROM models")}
    expected = {"tabulated_n", "tabulated_k"} | {f"formula_{i}" for i in range(1, 10)}
    extra = kinds - expected
    assert not extra, f"unexpected model kinds: {extra}"


def test_at_most_one_n_and_one_k_per_material(conn):
    rows = conn.execute(
        "SELECT shelf, book, page, quantity, COUNT(*)"
        " FROM models GROUP BY shelf, book, page, quantity"
        " HAVING COUNT(*) > 1"
    ).fetchall()
    assert rows == [], f"materials with duplicate quantity rows: {rows}"


def test_quantity_values(conn):
    quantities = {
        row[0] for row in conn.execute("SELECT DISTINCT quantity FROM models")
    }
    assert quantities == {"n", "k"}


def test_n_points_set_iff_tabulated(conn):
    bad = conn.execute(
        "SELECT model_kind, COUNT(*) FROM models"
        " WHERE (model_kind LIKE 'tabulated_%' AND n_points IS NULL)"
        "    OR (model_kind LIKE 'formula_%'  AND n_points IS NOT NULL)"
        " GROUP BY model_kind"
    ).fetchall()
    assert bad == [], bad


# --- numeric spot checks ------------------------------------------------

# Source coefficients copied verbatim from the upstream YAML files.
# Tolerances reflect float32 precision (~7 significant digits).
F32_REL_TOL = 1e-6


def _assert_floats_close(actual, expected, rel_tol: float = F32_REL_TOL) -> None:
    assert len(actual) == len(expected)
    for a, e in zip(actual, expected):
        assert math.isclose(a, e, rel_tol=rel_tol, abs_tol=1e-12), (a, e)


def test_sio2_malitson_formula1(conn):
    """main/SiO2/Malitson — Sellmeier formula 1, classic fused silica reference."""
    row = conn.execute(
        "SELECT model_kind, payload, wl_min, wl_max"
        " FROM models WHERE shelf=? AND book=? AND page=? AND quantity='n'",
        ("main", "SiO2", "Malitson"),
    ).fetchone()
    assert row is not None
    kind, payload, wmn, wmx = row
    assert kind == "formula_1"
    assert math.isclose(wmn, 0.21, rel_tol=F32_REL_TOL)
    assert math.isclose(wmx, 6.7, rel_tol=F32_REL_TOL)

    coeffs = decode_formula(payload)
    expected = (0.0, 0.6961663, 0.0684043, 0.4079426, 0.1162414, 0.8974794, 9.896161)
    _assert_floats_close(coeffs, expected)

    # Malitson provides only n, no k.
    k_row = conn.execute(
        "SELECT 1 FROM models WHERE shelf=? AND book=? AND page=? AND quantity='k'",
        ("main", "SiO2", "Malitson"),
    ).fetchone()
    assert k_row is None


def test_n_bk7_formula2_plus_tabulated_k(conn):
    """specs/SCHOTT-optical/N-BK7 — formula_2 for n, tabulated for k."""
    n_row = conn.execute(
        "SELECT model_kind, payload, wl_min, wl_max"
        " FROM models WHERE shelf=? AND book=? AND page=? AND quantity='n'",
        ("specs", "SCHOTT-optical", "N-BK7"),
    ).fetchone()
    assert n_row is not None
    kind, payload, wmn, wmx = n_row
    assert kind == "formula_2"
    assert math.isclose(wmn, 0.3, rel_tol=F32_REL_TOL)
    assert math.isclose(wmx, 2.5, rel_tol=F32_REL_TOL)
    coeffs = decode_formula(payload)
    expected = (
        0.0,
        1.03961212,
        0.00600069867,
        0.231792344,
        0.0200179144,
        1.01046945,
        103.560653,
    )
    _assert_floats_close(coeffs, expected)

    k_row = conn.execute(
        "SELECT model_kind, n_points, payload, wl_min, wl_max"
        " FROM models WHERE shelf=? AND book=? AND page=? AND quantity='k'",
        ("specs", "SCHOTT-optical", "N-BK7"),
    ).fetchone()
    assert k_row is not None
    kind, npts, payload, wmn, wmx = k_row
    assert kind == "tabulated_k"
    assert npts > 0
    wls, ks = decode_tabulated(payload, npts)
    assert len(wls) == npts
    assert math.isclose(min(wls), wmn, rel_tol=F32_REL_TOL)
    assert math.isclose(max(wls), wmx, rel_tol=F32_REL_TOL)
    # First sample from the source: 0.300  2.8607E-06
    assert math.isclose(wls[0], 0.300, rel_tol=F32_REL_TOL)
    assert math.isclose(ks[0], 2.8607e-06, rel_tol=1e-4)


def test_h2o_hale_tabulated_nk(conn):
    """main/H2O/Hale — `tabulated nk` upstream, split into both n and k rows."""
    rows = {
        q: (kind, npts, payload)
        for q, kind, npts, payload in conn.execute(
            "SELECT quantity, model_kind, n_points, payload"
            " FROM models WHERE shelf=? AND book=? AND page=?",
            ("main", "H2O", "Hale"),
        )
    }
    assert set(rows) == {"n", "k"}
    n_kind, n_pts, n_payload = rows["n"]
    k_kind, k_pts, k_payload = rows["k"]
    assert n_kind == "tabulated_n"
    assert k_kind == "tabulated_k"
    assert n_pts == k_pts > 0

    n_wls, n_vals = decode_tabulated(n_payload, n_pts)
    k_wls, k_vals = decode_tabulated(k_payload, k_pts)

    # Wavelength array is shared between n and k by construction.
    assert n_wls == k_wls

    # First source row: 0.200 1.396 1.10E-7
    assert math.isclose(n_wls[0], 0.200, rel_tol=F32_REL_TOL)
    assert math.isclose(n_vals[0], 1.396, rel_tol=F32_REL_TOL)
    assert math.isclose(k_vals[0], 1.10e-7, rel_tol=1e-4)

# --- catalog count guards -----------------------------------------------


def test_total_models_equals_sum_of_n_and_k(conn):
    n_only = conn.execute("SELECT COUNT(*) FROM models WHERE quantity='n'").fetchone()[
        0
    ]
    k_only = conn.execute("SELECT COUNT(*) FROM models WHERE quantity='k'").fetchone()[
        0
    ]
    total = conn.execute("SELECT COUNT(*) FROM models").fetchone()[0]
    assert n_only + k_only == total
