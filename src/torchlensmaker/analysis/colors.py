import torch
import colorcet as cc
import matplotlib as mpl

from typing import TypeAlias

from torchlensmaker.optical_data import OpticalData


# Color theme
color_valid = "#ffa724"
color_blocked = "red"
color_focal_point = "red"

# Default colormap*
LinearSegmentedColormap: TypeAlias = mpl.colors.LinearSegmentedColormap
default_colormap = cc.cm.CET_I2


Tensor: TypeAlias = torch.Tensor

def color_rays(
    data: OpticalData,
    color_dim: str,
    colormap: LinearSegmentedColormap = default_colormap,
) -> Tensor:

    color_tensor = data.get_rays(color_dim)

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
