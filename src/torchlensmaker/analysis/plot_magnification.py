import torch
import torch.nn as nn
import torchlensmaker as tlm


import matplotlib.pyplot as plt

from typing import Dict, Any, Optional

from torchlensmaker.analysis.colors import (
    LinearSegmentedColormap,
    default_colormap,
    color_valid,
    color_rays,
)


Tensor = torch.Tensor


def plot_magnification(
    optics: nn.Module,
    sampling: Dict[str, Any],
    color_dim: Optional[str] = None,
    colormap: LinearSegmentedColormap = default_colormap,
) -> None:
    """
    Compute and plot magnification data for the given optical system
    The system must compute object and image coordinates
    """

    # Evaluate the optical stack
    output = optics(tlm.default_input(sampling=sampling, dim=2, dtype=torch.float64))

    # Extract object and image coordinate (called T and V)
    T = output.rays_object
    V = output.rays_image

    mag, residuals = tlm.linear_magnification(T, V)

    # Get color data
    color_data = (
        color_rays(output, color_dim, colormap).tolist()
        if color_dim is not None
        else color_valid
    )

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(
        T.detach().numpy(),
        V.detach().numpy(),
        c=color_data,
        marker=".",
        s=10,
    )

    X = torch.linspace(T.min().item(), T.max().item(), 50)
    ax.plot(
        X.detach().numpy(),
        (mag * X).detach().numpy(),
        color="lightgrey",
        label=f"mag = {mag:.2f}",
    )

    ax.set_xlabel("Object coordinates")
    ax.set_ylabel("Image coordinates")
    ax.legend()

    plt.show()
