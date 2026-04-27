"""Dispersion-model dataclasses exposed by the public API.

Every dataclass is frozen and contains only its own data. None of them carries
any evaluation logic — that responsibility is left to consumers, who can build
whatever evaluator (numpy, jax, plain Python, symbolic, …) suits their needs.

Wavelengths are expressed in **micrometers** throughout the entire library, in
keeping with the upstream refractiveindex.info convention. The library performs
no unit conversion.

Tabulated arrays are stored as raw `bytes` containing **packed IEEE 754 float32
values in little-endian byte order**. This layout is part of the API contract.
Consumers decode with whichever tool they prefer:

    # numpy:
    import numpy as np
    wls = np.frombuffer(piece.wavelength_um, dtype=np.float32)

    # stdlib only:
    import array
    wls = array.array("f"); wls.frombytes(piece.wavelength_um)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WavelengthRange:
    """Closed wavelength interval over which a dispersion model is valid."""

    min_um: float
    max_um: float


# --- tabulated quantity ----------------------------------------------------


@dataclass(frozen=True)
class Tabulated:
    """Tabulated quantity sampled across wavelengths.

    Whether this describes n(λ) or k(λ) is determined by which slot of
    `MaterialEntry` it occupies (`entry.n` vs `entry.k`) — the dataclass
    itself is symmetric. `wavelength_um` and `values` are raw bytes
    containing `length` packed little-endian float32 values each. Linear
    interpolation is the upstream convention but the library does not
    perform it.
    """

    wavelength_um: bytes
    values: bytes
    length: int
    wavelength_range: WavelengthRange


# --- closed-form models for n(λ) -------------------------------------------
# All closed-form models share the upstream encoding "leading constant +
# pairs". The leading constant is held as `c1` (matching the 1-indexed naming
# of the upstream "Dispersion formulas" document); the pair-structured rest
# of the coefficient list lives in a `coefficients` or `terms` field.


@dataclass(frozen=True)
class Sellmeier:
    """Formula 1 — Sellmeier (preferred): n²−1 = C₁ + Σᵢ Bᵢ λ² / (λ² − Cᵢ²).

    `coefficients` is a tuple of `(B, C)` pairs. `c1` is the leading constant
    (often 0 in source data, occasionally non-zero). The naming matches the
    upstream "Dispersion formulas" document, which is 1-indexed.
    """

    c1: float
    coefficients: tuple[tuple[float, float], ...]
    wavelength_range: WavelengthRange


@dataclass(frozen=True)
class Sellmeier2:
    """Formula 2 — Sellmeier-2: n²−1 = C₁ + Σᵢ Bᵢ λ² / (λ² − Cᵢ).

    Differs from `Sellmeier` only in that the second member of each pair is
    not squared inside the denominator. Stored as `(B, C)` pairs.
    """

    c1: float
    coefficients: tuple[tuple[float, float], ...]
    wavelength_range: WavelengthRange


@dataclass(frozen=True)
class Polynomial:
    """Formula 3 — Polynomial: n² = C₁ + Σᵢ cᵢ λ^(eᵢ).

    `terms` is a tuple of `(coefficient, exponent)` pairs.
    """

    c1: float
    terms: tuple[tuple[float, float], ...]
    wavelength_range: WavelengthRange


@dataclass(frozen=True)
class RefractiveIndexInfoFormula4:
    """Formula 4 — RefractiveIndex.INFO mixed-form:

        n² = c1 + Σⱼ (Bⱼ λ^eBⱼ) / (λ² − Cⱼ^eCⱼ) + Σₖ (aₖ λ^eₖ)

    The upstream spec allows up to 2 rational terms and 4 polynomial pairs.
    In practice no upstream entry uses more than 3 polynomial pairs, and
    rational terms whose leading multiplier (Bⱼ) is zero are dropped at load
    time so that `rational_terms` and `polynomial_terms` reflect what the
    entry actually models.
    """

    c1: float
    rational_terms: tuple[tuple[float, float, float, float], ...]   # (B, eB, C, eC)
    polynomial_terms: tuple[tuple[float, float], ...]               # (coef, exp)
    wavelength_range: WavelengthRange


@dataclass(frozen=True)
class Cauchy:
    """Formula 5 — Cauchy: n = C₁ + Σᵢ cᵢ λ^(eᵢ).

    Same `(coef, exp)` pair shape as `Polynomial` but applied directly to n
    rather than n².
    """

    c1: float
    terms: tuple[tuple[float, float], ...]
    wavelength_range: WavelengthRange


@dataclass(frozen=True)
class Gases:
    """Formula 6 — Gases: n−1 = C₁ + Σᵢ Bᵢ / (Cᵢ − λ⁻²).

    Used for gases at standard reference conditions (the upstream entries
    document the temperature/pressure in the per-entry `comments`).
    """

    c1: float
    coefficients: tuple[tuple[float, float], ...]
    wavelength_range: WavelengthRange


@dataclass(frozen=True)
class Herzberger:
    """Formula 7 — Herzberger:

        n = c1 + c2/(λ²−0.028) + c3 (1/(λ²−0.028))² + c4 λ² + c5 λ⁴ + c6 λ⁶

    Fixed shape with exactly six coefficients.
    """

    c1: float
    c2: float
    c3: float
    c4: float
    c5: float
    c6: float
    wavelength_range: WavelengthRange


@dataclass(frozen=True)
class Retro:
    """Formula 8 — Retro: (n²−1)/(n²+2) = c1 + c2 λ²/(λ²−c3) + c4 λ².

    Fixed shape with exactly four coefficients.
    """

    c1: float
    c2: float
    c3: float
    c4: float
    wavelength_range: WavelengthRange


@dataclass(frozen=True)
class Exotic:
    """Formula 9 — Exotic: n² = c1 + c2/(λ²−c3) + c4 (λ−c5) / ((λ−c5)² + c6).

    Fixed shape with exactly six coefficients.
    """

    c1: float
    c2: float
    c3: float
    c4: float
    c5: float
    c6: float
    wavelength_range: WavelengthRange


# --- unions & material entry -----------------------------------------------

NModel = (
    Tabulated
    | Sellmeier
    | Sellmeier2
    | Polynomial
    | RefractiveIndexInfoFormula4
    | Cauchy
    | Gases
    | Herzberger
    | Retro
    | Exotic
)

# The current upstream has no formula-based k descriptions; KModel collapses
# to Tabulated. If formula-based k descriptions ever appear upstream, the
# relevant formula types can be added to this union.
KModel = Tabulated


@dataclass(frozen=True)
class MaterialEntry:
    """A single material entry from the refractiveindex.info database.

    `n` and `k` are independently optional — the upstream may describe only
    one of the two quantities. If both are present they describe the real
    and imaginary parts of the complex refractive index over (potentially
    different) wavelength ranges.
    """

    shelf: str
    book: str
    page: str
    name: str | None
    references: str | None
    comments: str | None
    n: NModel | None
    k: KModel | None
