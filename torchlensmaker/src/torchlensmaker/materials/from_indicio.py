import torch
from indicio.models import Cauchy, Sellmeier, Sellmeier2, Tabulated

from .material_elements import (
    CauchyMaterial,
    LinearSegmentedMaterial,
    MaterialModel,
    SellmeierMaterial,
)


def from_indicio_tabulated(model: Tabulated) -> LinearSegmentedMaterial:
    wavelengths_nm = (
        torch.frombuffer(bytearray(model.wavelength_um), dtype=torch.float32).clone()
        * 1000.0
    )
    indices = torch.frombuffer(bytearray(model.values), dtype=torch.float32).clone()
    return LinearSegmentedMaterial(wavelengths_nm, indices)


def from_indicio_cauchy(model: Cauchy) -> CauchyMaterial:
    exponent_map = {-2.0: "B", -4.0: "C", -6.0: "D"}
    coeffs = {"B": 0.0, "C": 0.0, "D": 0.0}
    for coef, exp in model.terms:
        if exp not in exponent_map:
            raise ValueError(
                f"Cauchy term with exponent {exp} cannot be represented "
                f"in CauchyMaterial (supported exponents: -2, -4, -6)"
            )
        coeffs[exponent_map[exp]] = coef
    return CauchyMaterial(model.c1, coeffs["B"], coeffs["C"], coeffs["D"])


def _sellmeier_coefficients_to_material(
    c1: float,
    coefficients: tuple,
    square_c: bool,
) -> SellmeierMaterial:
    if c1 != 0.0:
        raise ValueError(
            f"Sellmeier model has non-zero leading constant c1={c1}, "
            f"which is not supported by SellmeierMaterial"
        )
    if len(coefficients) > 3:
        raise ValueError(
            f"Sellmeier model has {len(coefficients)} terms, "
            f"but SellmeierMaterial supports at most 3"
        )
    padded = list(coefficients) + [(0.0, 1.0)] * (3 - len(coefficients))
    b1, c1_ = padded[0]
    b2, c2 = padded[1]
    b3, c3 = padded[2]
    if square_c:
        c1_, c2, c3 = c1_**2, c2**2, c3**2
    return SellmeierMaterial(b1, b2, b3, c1_, c2, c3)


def from_indicio_sellmeier(model: Sellmeier) -> SellmeierMaterial:
    # Formula 1: denominator is (λ² - Cᵢ²), so tlm Cᵢ = indicio Cᵢ²
    return _sellmeier_coefficients_to_material(
        model.c1, model.coefficients, square_c=True
    )


def from_indicio_sellmeier2(model: Sellmeier2) -> SellmeierMaterial:
    # Formula 2: denominator is (λ² - Cᵢ), direct mapping
    return _sellmeier_coefficients_to_material(
        model.c1, model.coefficients, square_c=False
    )


def material_from_indicio(entry) -> MaterialModel:
    """Convert an indicio.MaterialEntry to the appropriate MaterialModel subtype.

    Supports Tabulated, Cauchy (Formula 5), Sellmeier (Formula 1), and
    Sellmeier2 (Formula 2) models. Raises NotImplementedError for other types.
    """
    if entry.n is None:
        raise ValueError("MaterialEntry has no refractive index model (n is None)")

    match entry.n:
        case Tabulated():
            return from_indicio_tabulated(entry.n)
        case Cauchy():
            return from_indicio_cauchy(entry.n)
        case Sellmeier():
            return from_indicio_sellmeier(entry.n)
        case Sellmeier2():
            return from_indicio_sellmeier2(entry.n)
        case _:
            raise NotImplementedError(
                f"Unsupported indicio model type: {type(entry.n).__name__}. "
                f"Supported types: Tabulated, Cauchy, Sellmeier, Sellmeier2."
            )
