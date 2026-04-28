"""End-to-end tests for the public API: loader → dataclass mapping plus
lookup/browse ergonomics. These tests go through `indicio` proper rather
than reaching into SQLite directly — they exercise everything the SQLite-
level tests in `test_database.py` don't cover.
"""

from __future__ import annotations

import array
import math

import pytest

import indicio
from indicio.models import (
    Cauchy,
    Exotic,
    Gases,
    Herzberger,
    Polynomial,
    RefractiveIndexInfoFormula4,
    Retro,
    Sellmeier,
    Sellmeier2,
    Tabulated,
)


F32_REL_TOL = 1e-6


# --- loader → dataclass mapping ----------------------------------------


def test_sio2_malitson_maps_to_sellmeier():
    """formula_1 → Sellmeier with (B, C) pairs in upstream order."""
    entry = indicio.get_material("main", "SiO2", "Malitson")
    assert entry.k is None
    assert isinstance(entry.n, Sellmeier)
    assert math.isclose(entry.n.c1, 0.0, abs_tol=1e-12)
    expected_pairs = (
        (0.6961663, 0.0684043),
        (0.4079426, 0.1162414),
        (0.8974794, 9.896161),
    )
    assert len(entry.n.coefficients) == 3
    for got, want in zip(entry.n.coefficients, expected_pairs):
        assert math.isclose(got[0], want[0], rel_tol=F32_REL_TOL)
        assert math.isclose(got[1], want[1], rel_tol=F32_REL_TOL)
    assert math.isclose(entry.n.wavelength_range[0], 0.21, rel_tol=F32_REL_TOL)
    assert math.isclose(entry.n.wavelength_range[1], 6.7, rel_tol=F32_REL_TOL)


def test_sio2_malitson_evaluates_to_textbook_value():
    """Full-stack check: the public dataclass evaluates correctly when
    consumers plug it into the canonical Sellmeier formula."""
    entry = indicio.get_material("main", "SiO2", "Malitson")
    m = entry.n
    assert isinstance(m, Sellmeier)
    lam = 0.5876  # sodium D line
    n2m1 = m.c1 + sum(B * lam**2 / (lam**2 - C * C) for B, C in m.coefficients)
    assert math.isclose(math.sqrt(1 + n2m1), 1.45846, abs_tol=1e-4)


def test_agase2_boyd_o_implicit_zero_pairing():
    """formula_2 with 4 coefficients — odd one out, the loader pads the
    trailing pair member to 0 so the dataclass shape stays uniform."""
    entry = indicio.get_material("main", "AgGaSe2", "Boyd-o")
    assert isinstance(entry.n, Sellmeier2)
    assert math.isclose(entry.n.c1, 3.6453, rel_tol=F32_REL_TOL)
    assert len(entry.n.coefficients) == 2
    (B1, C1), (B2, C2) = entry.n.coefficients
    assert math.isclose(B1, 2.2057, rel_tol=F32_REL_TOL)
    assert math.isclose(C1, 0.1879, rel_tol=F32_REL_TOL)
    assert math.isclose(B2, 1.8377, rel_tol=F32_REL_TOL)
    assert C2 == 0.0  # implicit zero pad


def test_si_edwards_herzberger_zero_pad():
    """formula_7 — only entry has 5 coefficients; loader pads c6 to 0."""
    entry = indicio.get_material("main", "Si", "Edwards")
    assert isinstance(entry.n, Herzberger)
    assert math.isclose(entry.n.c1, 3.41983, rel_tol=F32_REL_TOL)
    assert math.isclose(entry.n.c2, 0.159906, rel_tol=F32_REL_TOL)
    assert math.isclose(entry.n.c3, -0.123109, rel_tol=F32_REL_TOL)
    assert math.isclose(entry.n.c4, 1.26878e-6, rel_tol=F32_REL_TOL)
    assert math.isclose(entry.n.c5, -1.95104e-9, rel_tol=F32_REL_TOL)
    assert entry.n.c6 == 0.0
    assert entry.n.wavelength_range == pytest.approx((2.4373, 25.0), rel=F32_REL_TOL)


def test_baf2_bosomworth_formula4_two_rationals():
    """formula_4 — both rational slots non-zero, no polynomial pairs."""
    entry = indicio.get_material("main", "BaF2", "Bosomworth-300K")
    assert isinstance(entry.n, RefractiveIndexInfoFormula4)
    assert entry.n.c1 == 0.0
    assert len(entry.n.rational_terms) == 2
    B1, eB1, C1, eC1 = entry.n.rational_terms[0]
    assert math.isclose(B1, 6.94, rel_tol=F32_REL_TOL)
    assert eB1 == 2.0
    assert math.isclose(C1, 54.348, rel_tol=1e-5)
    assert eC1 == 2.0
    B2, _, _, _ = entry.n.rational_terms[1]
    assert math.isclose(B2, -6350.4, rel_tol=1e-5)
    assert entry.n.polynomial_terms == ()


def test_au_johnson_tabulated_layout():
    """Tabulated payload: `wavelength_um` and `values` blobs are each
    `length * 4` bytes; the wavelength range in the dataclass agrees with
    the first/last decoded sample."""
    entry = indicio.get_material("main", "Au", "Johnson")
    assert isinstance(entry.n, Tabulated)
    assert isinstance(entry.k, Tabulated)
    n = entry.n
    assert n.length > 0
    assert len(n.wavelength_um) == n.length * 4
    assert len(n.values) == n.length * 4
    wls = array.array("f")
    wls.frombytes(n.wavelength_um)
    assert math.isclose(wls[0], n.wavelength_range[0], rel_tol=F32_REL_TOL)
    assert math.isclose(wls[-1], n.wavelength_range[1], rel_tol=F32_REL_TOL)


def test_h2o_hale_tabulated_nk_shares_wavelength_bytes():
    """`tabulated nk` entries are split into separate Tabulated rows that
    duplicate the wavelength array verbatim."""
    entry = indicio.get_material("main", "H2O", "Hale")
    assert isinstance(entry.n, Tabulated)
    assert isinstance(entry.k, Tabulated)
    assert entry.n.length == entry.k.length
    assert entry.n.wavelength_um == entry.k.wavelength_um
    assert entry.n.values != entry.k.values  # different quantities


# --- public API ergonomics ---------------------------------------------


def test_get_material_raises_keyerror_on_miss():
    with pytest.raises(KeyError) as excinfo:
        indicio.get_material("main", "SiO2", "DoesNotExist")
    assert excinfo.value.args[0] == ("main", "SiO2", "DoesNotExist")


def test_has_material_true_and_false():
    assert indicio.has_material("main", "SiO2", "Malitson") is True
    assert indicio.has_material("main", "SiO2", "DoesNotExist") is False
    assert indicio.has_material("nope", "nope", "nope") is False


def test_shelves_sorted_and_deduplicated():
    s = indicio.shelves()
    assert isinstance(s, tuple)
    assert len(s) > 0
    assert list(s) == sorted(s)
    assert len(set(s)) == len(s)
    # well-known shelves we expect upstream to keep stable
    assert "main" in s
    assert "glass" in s


def test_books_sorted_and_deduplicated():
    b = indicio.books("main")
    assert isinstance(b, tuple)
    assert len(b) > 0
    assert list(b) == sorted(b)
    assert len(set(b)) == len(b)
    assert "SiO2" in b


def test_books_unknown_shelf_returns_empty():
    assert indicio.books("does-not-exist") == ()


def test_pages_sorted_and_deduplicated():
    p = indicio.pages("main", "SiO2")
    assert isinstance(p, tuple)
    assert len(p) > 0
    assert list(p) == sorted(p)
    assert len(set(p)) == len(p)
    assert "Malitson" in p


def test_pages_unknown_book_returns_empty():
    assert indicio.pages("main", "does-not-exist") == ()


def test_search_finds_known_entry():
    hits = indicio.search("Malitson")
    assert ("main", "SiO2", "Malitson") in hits
    # hits are sorted lexicographically by (shelf, book, page)
    assert list(hits) == sorted(hits)


def test_search_no_match_returns_empty():
    hits = indicio.search("zzz-no-such-material-zzz")
    assert hits == ()


def test_iter_materials_yields_every_entry():
    """Stream count equals the materials-table count (no dropped rows)."""
    streamed = sum(1 for _ in indicio.iter_materials())
    expected = sum(len(indicio.pages(s, b)) for s in indicio.shelves() for b in indicio.books(s))
    assert streamed == expected
