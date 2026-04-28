import struct

import indicio
import pytest
import torch
from indicio.models import (
    Cauchy,
    MaterialEntry,
    Polynomial,
    Sellmeier,
    Sellmeier2,
    Tabulated,
)

from torchlensmaker.materials.from_indicio import material_from_indicio
from torchlensmaker.materials.material_elements import (
    CauchyMaterial,
    LinearSegmentedMaterial,
    SellmeierMaterial,
)


def make_entry(n_model):
    return MaterialEntry(
        shelf="test",
        book="test",
        page="test",
        name=None,
        references=None,
        comments=None,
        n=n_model,
        k=None,
    )


def pack_floats(values):
    return struct.pack(f"{len(values)}f", *values)


# --- Tabulated ---------------------------------------------------------------


def test_tabulated_returns_linear_segmented():
    wl_um = [0.4, 0.5, 0.6, 0.7]
    n_vals = [1.52, 1.51, 1.50, 1.49]
    tab = Tabulated(
        wavelength_um=pack_floats(wl_um),
        values=pack_floats(n_vals),
        length=4,
        wavelength_range=(0.4, 0.7),
    )
    model = material_from_indicio(make_entry(tab))
    assert isinstance(model, LinearSegmentedMaterial)


def test_tabulated_wavelengths_converted_to_nm():
    wl_um = [0.4, 0.5, 0.6, 0.7]
    n_vals = [1.52, 1.51, 1.50, 1.49]
    tab = Tabulated(
        wavelength_um=pack_floats(wl_um),
        values=pack_floats(n_vals),
        length=4,
        wavelength_range=(0.4, 0.7),
    )
    model = material_from_indicio(make_entry(tab))
    expected_nm = torch.tensor([400.0, 500.0, 600.0, 700.0], dtype=torch.float32)
    assert torch.allclose(model.wavelengths, expected_nm, atol=0.1)


def test_tabulated_indices_preserved():
    wl_um = [0.4, 0.5, 0.6]
    n_vals = [1.52, 1.51, 1.50]
    tab = Tabulated(
        wavelength_um=pack_floats(wl_um),
        values=pack_floats(n_vals),
        length=3,
        wavelength_range=(0.4, 0.6),
    )
    model = material_from_indicio(make_entry(tab))
    expected = torch.tensor(n_vals, dtype=torch.float32)
    assert torch.allclose(model.indices, expected, atol=1e-5)


# --- Cauchy ------------------------------------------------------------------


def test_cauchy_returns_cauchy_material():
    cauchy = Cauchy(c1=1.5, terms=((0.003, -2.0),), wavelength_range=(0.4, 0.7))
    model = material_from_indicio(make_entry(cauchy))
    assert isinstance(model, CauchyMaterial)


def test_cauchy_maps_coefficients():
    cauchy = Cauchy(
        c1=1.5,
        terms=((0.003, -2.0), (0.0001, -4.0), (1e-6, -6.0)),
        wavelength_range=(0.4, 0.7),
    )
    model = material_from_indicio(make_entry(cauchy))
    assert model.A.item() == pytest.approx(1.5)
    assert model.B.item() == pytest.approx(0.003)
    assert model.C.item() == pytest.approx(0.0001)
    assert model.D.item() == pytest.approx(1e-6)


def test_cauchy_missing_terms_default_to_zero():
    cauchy = Cauchy(c1=1.5, terms=((0.003, -2.0),), wavelength_range=(0.4, 0.7))
    model = material_from_indicio(make_entry(cauchy))
    assert model.C.item() == pytest.approx(0.0)
    assert model.D.item() == pytest.approx(0.0)


def test_cauchy_non_standard_exponent_raises():
    cauchy = Cauchy(c1=1.5, terms=((0.003, 2.0),), wavelength_range=(0.4, 0.7))
    with pytest.raises(ValueError, match="exponent"):
        material_from_indicio(make_entry(cauchy))


# --- Sellmeier (Formula 1 — Cᵢ squared in denominator) ---------------------


def test_sellmeier_returns_sellmeier_material():
    s = Sellmeier(c1=0.0, coefficients=((1.0, 0.1),), wavelength_range=(0.4, 0.7))
    model = material_from_indicio(make_entry(s))
    assert isinstance(model, SellmeierMaterial)


def test_sellmeier_c_values_are_squared():
    B, C = 0.7, 0.068  # typical quartz-like values
    s = Sellmeier(c1=0.0, coefficients=((B, C),), wavelength_range=(0.4, 0.7))
    model = material_from_indicio(make_entry(s))
    assert model.B1.item() == pytest.approx(B)
    assert model.C1.item() == pytest.approx(C**2)


def test_sellmeier_pads_to_three_terms():
    s = Sellmeier(c1=0.0, coefficients=((1.0, 0.1),), wavelength_range=(0.4, 0.7))
    model = material_from_indicio(make_entry(s))
    assert model.B2.item() == pytest.approx(0.0)
    assert model.B3.item() == pytest.approx(0.0)


def test_sellmeier_nonzero_c1_raises():
    s = Sellmeier(c1=0.5, coefficients=((1.0, 0.1),), wavelength_range=(0.4, 0.7))
    with pytest.raises(ValueError, match="c1"):
        material_from_indicio(make_entry(s))


def test_sellmeier_more_than_three_terms_raises():
    coeffs = tuple((float(i), 0.01 * i) for i in range(1, 5))
    s = Sellmeier(c1=0.0, coefficients=coeffs, wavelength_range=(0.4, 0.7))
    with pytest.raises(ValueError, match="3"):
        material_from_indicio(make_entry(s))


# --- Sellmeier2 (Formula 2 — direct mapping) ---------------------------------


def test_sellmeier2_returns_sellmeier_material():
    s2 = Sellmeier2(c1=0.0, coefficients=((1.0, 0.01),), wavelength_range=(0.4, 0.7))
    model = material_from_indicio(make_entry(s2))
    assert isinstance(model, SellmeierMaterial)


def test_sellmeier2_c_values_direct():
    B, C = 1.04, 0.006
    s2 = Sellmeier2(c1=0.0, coefficients=((B, C),), wavelength_range=(0.4, 0.7))
    model = material_from_indicio(make_entry(s2))
    assert model.B1.item() == pytest.approx(B)
    assert model.C1.item() == pytest.approx(C)  # not squared


def test_sellmeier2_nonzero_c1_raises():
    s2 = Sellmeier2(c1=0.1, coefficients=((1.0, 0.01),), wavelength_range=(0.4, 0.7))
    with pytest.raises(ValueError, match="c1"):
        material_from_indicio(make_entry(s2))


# --- Unsupported types -------------------------------------------------------


def test_unsupported_type_raises():
    poly = Polynomial(c1=1.0, terms=((0.5, 2.0),), wavelength_range=(0.4, 0.7))
    with pytest.raises(NotImplementedError):
        material_from_indicio(make_entry(poly))


def test_none_n_raises():
    entry = MaterialEntry(
        shelf="test",
        book="test",
        page="test",
        name=None,
        references=None,
        comments=None,
        n=None,
        k=None,
    )
    with pytest.raises(ValueError):
        material_from_indicio(entry)


# --- Integration test with real indicio data ---------------------------------


def test_bk7_schott_values():
    entry = indicio.get_material("popular_glass", "BK7", "SCHOTT")
    model = material_from_indicio(entry)
    assert isinstance(model, SellmeierMaterial)
    wl = torch.tensor([486.0, 589.0, 656.0])
    n = model(wl)
    assert n[0].item() == pytest.approx(1.5224, abs=5e-4)
    assert n[1].item() == pytest.approx(1.5168, abs=5e-4)
    assert n[2].item() == pytest.approx(1.5143, abs=5e-4)
