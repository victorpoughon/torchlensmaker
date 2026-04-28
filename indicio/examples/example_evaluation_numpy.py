"""Evaluate a material's n(λ) and k(λ) using numpy.

Each evaluator is vectorized: pass a numpy array of wavelengths and get an
array back in one call. The library itself stays numpy-free; numpy is only
used here on the consumer side.

Usage:
    uv run python examples/example_evaluation_numpy.py <shelf> <book> <page>

Example:
    uv run python examples/example_evaluation_numpy.py main SiO2 Malitson
"""

from __future__ import annotations

import sys

import numpy as np

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


def evaluate_tabulated(t: Tabulated, lam: np.ndarray) -> np.ndarray:
    wls = np.frombuffer(t.wavelength_um, dtype=np.float32)
    vals = np.frombuffer(t.values, dtype=np.float32)
    return np.interp(lam, wls, vals)


def evaluate_sellmeier(m: Sellmeier, lam: np.ndarray) -> np.ndarray:
    n2m1 = m.c1 + sum(B * lam**2 / (lam**2 - C * C) for B, C in m.coefficients)
    return np.sqrt(1.0 + n2m1)


def evaluate_sellmeier2(m: Sellmeier2, lam: np.ndarray) -> np.ndarray:
    n2m1 = m.c1 + sum(B * lam**2 / (lam**2 - C) for B, C in m.coefficients)
    return np.sqrt(1.0 + n2m1)


def evaluate_polynomial(m: Polynomial, lam: np.ndarray) -> np.ndarray:
    return np.sqrt(m.c1 + sum(coef * lam**exp for coef, exp in m.terms))


def evaluate_formula4(m: RefractiveIndexInfoFormula4, lam: np.ndarray) -> np.ndarray:
    n2 = np.full_like(lam, m.c1, dtype=np.float64)
    for B, eB, C, eC in m.rational_terms:
        n2 = n2 + B * lam**eB / (lam**2 - C**eC)
    for coef, exp in m.polynomial_terms:
        n2 = n2 + coef * lam**exp
    return np.sqrt(n2)


def evaluate_cauchy(m: Cauchy, lam: np.ndarray) -> np.ndarray:
    return np.array(m.c1 + sum(coef * lam**exp for coef, exp in m.terms))


def evaluate_gases(m: Gases, lam: np.ndarray) -> np.ndarray:
    return np.array(1.0 + m.c1 + sum(B / (C - lam**-2) for B, C in m.coefficients))


def evaluate_herzberger(m: Herzberger, lam: np.ndarray) -> np.ndarray:
    inv = 1.0 / (lam**2 - 0.028)
    return (
        m.c1
        + m.c2 * inv
        + m.c3 * inv * inv
        + m.c4 * lam**2
        + m.c5 * lam**4
        + m.c6 * lam**6
    )


def evaluate_retro(m: Retro, lam: np.ndarray) -> np.ndarray:
    rhs = m.c1 + m.c2 * lam**2 / (lam**2 - m.c3) + m.c4 * lam**2
    return np.sqrt((1.0 + 2.0 * rhs) / (1.0 - rhs))


def evaluate_exotic(m: Exotic, lam: np.ndarray) -> np.ndarray:
    n2 = (
        m.c1 + m.c2 / (lam**2 - m.c3) + m.c4 * (lam - m.c5) / ((lam - m.c5) ** 2 + m.c6)
    )
    return np.sqrt(n2)


def evaluate(model: object, lam: np.ndarray) -> np.ndarray:
    match model:
        case Sellmeier():
            return evaluate_sellmeier(model, lam)
        case Sellmeier2():
            return evaluate_sellmeier2(model, lam)
        case Polynomial():
            return evaluate_polynomial(model, lam)
        case RefractiveIndexInfoFormula4():
            return evaluate_formula4(model, lam)
        case Cauchy():
            return evaluate_cauchy(model, lam)
        case Gases():
            return evaluate_gases(model, lam)
        case Herzberger():
            return evaluate_herzberger(model, lam)
        case Retro():
            return evaluate_retro(model, lam)
        case Exotic():
            return evaluate_exotic(model, lam)
        case Tabulated():
            return evaluate_tabulated(model, lam)
    raise TypeError(f"unknown model type: {type(model).__name__}")


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        print(__doc__, file=sys.stderr)
        return 2
    _, shelf, book, page = argv
    entry = indicio.get_material(shelf, book, page)

    print(f"{entry.shelf}/{entry.book}/{entry.page}: {entry.name}")
    print(f"  n = {type(entry.n).__name__ if entry.n else None}")
    print(f"  k = {type(entry.k).__name__ if entry.k else None}")

    if entry.n is not None:
        lo, hi = entry.n.wavelength_range
        wls = np.linspace(lo, hi, 5)
        ns = evaluate(entry.n, wls)
        print(f"\nn(λ) over [{lo}, {hi}] µm:")
        for lam, val in zip(wls, ns):
            print(f"  λ = {lam:>8.4f} µm   n = {val:.6f}")

    if entry.k is not None:
        lo, hi = entry.k.wavelength_range
        wls = np.linspace(lo, hi, 5)
        ks = evaluate(entry.k, wls)
        print(f"\nk(λ) over [{lo}, {hi}] µm:")
        for lam, val in zip(wls, ks):
            print(f"  λ = {lam:>8.4f} µm   k = {val:.6e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
