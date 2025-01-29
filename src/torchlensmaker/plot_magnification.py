import torch
import torch.nn as nn
import torchlensmaker as tlm

import matplotlib as mpl
import matplotlib.pyplot as plt

from typing import Dict, Any, Optional

from torchlensmaker.viewer.render_sequence import (
    default_colormap,
    color_rays,
    color_valid,
)


def plot_magnification(
    optics: nn.Module,
    sampling: Dict[str, Any],
    color_dim: Optional[str] = None,
    colormap: mpl.colors.LinearSegmentedColormap = default_colormap,
):
    """
    Compute and plot magnification data for the given optical system
    The system must compute object and image coordinates
    """

    # Evaluate the optical stack
    output = optics(tlm.default_input(dim=2, dtype=torch.float64, sampling=sampling))

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
