import torch
import torchlensmaker as tlm

import matplotlib.pyplot as plt

from typing import Optional


def plot_material_models(
    models: list[tlm.MaterialModel],
    wmin: float | int = 400,
    wmax: float | int = 700,
    labels: Optional[list[str]] = None,
) -> None:
    """
    Plot multiple dispersion models from wmin to wmax (in nm)

    Args:
        models: list of DispersionModel objects
        labels: optional list of labels
    """
    W = torch.linspace(wmin, wmax, 1000)

    fig, ax = plt.subplots(figsize=(12, 8))

    if labels is None:
        labels = [type(m) for m in models]

    for label, model in zip(labels, models):
        N = model.refractive_index(W)
        ax.plot(W, N, label=label)

    ax.legend()
    ax.set_ylabel("Index of refraction")
    ax.set_xlabel("Wavelength (nm)")
