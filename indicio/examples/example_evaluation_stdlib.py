"""Evaluate a material's n(λ) and k(λ) using only the Python standard library.

Usage:
    uv run python examples/example_evaluation_stdlib.py <shelf> <book> <page>

Example:
    uv run python examples/example_evaluation_stdlib.py main SiO2 Malitson
"""

from __future__ import annotations

import array
import bisect
import math
import sys

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


def evaluate_tabulated(t: Tabulated, lam: float) -> float:
    lo, hi = t.wavelength_range
    if not lo <= lam <= hi:
        raise ValueError(f"{lam} µm outside [{lo}, {hi}]")
    wls = array.array("f")
    wls.frombytes(t.wavelength_um)
    vals = array.array("f")
    vals.frombytes(t.values)
    i = bisect.bisect_left(wls, lam)
    if i == 0:
        return vals[0]
    if i == len(wls):
        return vals[-1]
    w = (lam - wls[i - 1]) / (wls[i] - wls[i - 1])
    return vals[i - 1] * (1 - w) + vals[i] * w


def evaluate_sellmeier(m: Sellmeier, lam: float) -> float:
    n2m1 = m.c1 + sum(B * lam**2 / (lam**2 - C * C) for B, C in m.coefficients)
    return math.sqrt(1.0 + n2m1)


def evaluate_sellmeier2(m: Sellmeier2, lam: float) -> float:
    n2m1 = m.c1 + sum(B * lam**2 / (lam**2 - C) for B, C in m.coefficients)
    return math.sqrt(1.0 + n2m1)


def evaluate_polynomial(m: Polynomial, lam: float) -> float:
    return math.sqrt(m.c1 + sum(coef * lam**exp for coef, exp in m.terms))


def evaluate_formula4(m: RefractiveIndexInfoFormula4, lam: float) -> float:
    n2 = m.c1
    for B, eB, C, eC in m.rational_terms:
        n2 += B * lam**eB / (lam**2 - C**eC)
    for coef, exp in m.polynomial_terms:
        n2 += coef * lam**exp
    return math.sqrt(n2)


def evaluate_cauchy(m: Cauchy, lam: float) -> float:
    return m.c1 + sum(coef * lam**exp for coef, exp in m.terms)


def evaluate_gases(m: Gases, lam: float) -> float:
    return 1.0 + m.c1 + sum(B / (C - lam**-2) for B, C in m.coefficients)


def evaluate_herzberger(m: Herzberger, lam: float) -> float:
    inv = 1.0 / (lam**2 - 0.028)
    return (
        m.c1
        + m.c2 * inv
        + m.c3 * inv * inv
        + m.c4 * lam**2
        + m.c5 * lam**4
        + m.c6 * lam**6
    )


def evaluate_retro(m: Retro, lam: float) -> float:
    rhs = m.c1 + m.c2 * lam**2 / (lam**2 - m.c3) + m.c4 * lam**2
    return math.sqrt((1.0 + 2.0 * rhs) / (1.0 - rhs))


def evaluate_exotic(m: Exotic, lam: float) -> float:
    n2 = (
        m.c1 + m.c2 / (lam**2 - m.c3) + m.c4 * (lam - m.c5) / ((lam - m.c5) ** 2 + m.c6)
    )
    return math.sqrt(n2)


def evaluate(model: object, lam: float) -> float:
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


def sample_points(lo: float, hi: float, count: int = 5) -> list[float]:
    if count == 1:
        return [(lo + hi) / 2]
    step = (hi - lo) / (count - 1)
    return [lo + i * step for i in range(count)]


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
        print(f"\nn(λ) over [{lo}, {hi}] µm:")
        for lam in sample_points(lo, hi):
            print(f"  λ = {lam:>8.4f} µm   n = {evaluate(entry.n, lam):.6f}")

    if entry.k is not None:
        lo, hi = entry.k.wavelength_range
        print(f"\nk(λ) over [{lo}, {hi}] µm:")
        for lam in sample_points(lo, hi):
            print(f"  λ = {lam:>8.4f} µm   k = {evaluate(entry.k, lam):.6e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
