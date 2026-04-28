"""Plot n(λ) and (when present) k(λ) for a single material.

Uses numpy for evaluation and matplotlib for the plot. Reuses the vectorized
evaluator dispatcher from `example_evaluation_numpy` so this script focuses
on plotting concerns only.

Usage:
    uv run python examples/example_plot.py <shelf> <book> <page>

Example:
    uv run python examples/example_plot.py main SiO2 Malitson
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import indicio

sys.path.insert(0, str(Path(__file__).resolve().parent))
from example_evaluation_numpy import evaluate  # noqa: E402

SAMPLES = 400
COLORS = {"n": "#AA4643", "k": "#4572A7"}


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        print(__doc__, file=sys.stderr)
        return 2
    _, shelf, book, page = argv
    entry = indicio.get_material(shelf, book, page)

    panels = [(q, m) for q, m in (("n", entry.n), ("k", entry.k)) if m is not None]
    if not panels:
        print(f"{shelf}/{book}/{page} has no n or k data", file=sys.stderr)
        return 1

    fig, axes = plt.subplots(len(panels), 1, figsize=(8, 3 * len(panels)), sharex=False)
    if len(panels) == 1:
        axes = [axes]

    for ax, (quantity, model) in zip(axes, panels):
        lo, hi = model.wavelength_range
        wls = np.linspace(lo, hi, SAMPLES)
        values = evaluate(model, wls)
        ax.plot(wls, values, color=COLORS[quantity])
        ax.set_xlabel("wavelength (µm)")
        ax.set_ylabel(quantity)
        ax.set_title(f"{quantity}(λ)  —  {type(model).__name__}")
        ax.grid(True, alpha=0.3)
        if quantity == "k":
            # k spans many orders of magnitude for most materials.
            if np.all(values > 0):
                ax.set_yscale("log")

    title = f"{entry.shelf}/{entry.book}/{entry.page}"
    if entry.name:
        title += f"  —  {entry.name}"
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
