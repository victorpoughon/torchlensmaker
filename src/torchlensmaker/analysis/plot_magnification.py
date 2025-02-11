import torch
import torch.nn as nn
import torchlensmaker as tlm


import matplotlib.pyplot as plt

from typing import Dict, Any, Optional

from torchlensmaker.analysis.colors import (
    LinearSegmentedColormap,
    default_colormap,
    color_valid,
)


Tensor = torch.Tensor


def color_rays_tensor(data: tlm.OpticalData, color_dim: str) -> Tensor:
    if color_dim == "base" and data.rays_base is not None:
        return data.rays_base
    elif color_dim == "object" and data.rays_object is not None:
        return data.rays_object
    elif color_dim == "wavelength" and data.rays_wavelength is not None:
        return data.rays_wavelength
    # TODO check that returned tensor is not None?
    else:
        raise RuntimeError(f"Unknown are unavailable color dimension '{color_dim}'")


def color_rays(
    data: tlm.OpticalData,
    color_dim: str,
    colormap: LinearSegmentedColormap = default_colormap,
) -> Tensor:

    color_tensor = color_rays_tensor(data, color_dim)

    # unsqueeze to 2D
    if color_tensor.dim() == 1:
        color_tensor = color_tensor.unsqueeze(1)

    assert color_tensor.dim() == 2
    assert color_tensor.shape[1] in {1, 2}

    # Ray variables that we use for coloring can be 2D when simulating in 3D
    # TODO more configurability here

    if color_tensor.shape[1] == 1:
        var = color_tensor[:, 0]
    else:
        # TODO 2D colormap?
        var = torch.linalg.vector_norm(color_tensor, dim=1)

    # normalize color variable to [0, 1]
    # unless the data range is too small, then use 0.5
    denom = var.max() - var.min()
    if denom > 1e-4:
        c = (var - var.min()) / denom
    else:
        c = torch.full_like(var, 0.5)

    # convert to rgb using color map
    return torch.tensor(colormap(c))



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
