# This file is part of Torch Lens Maker
# Copyright (C) 2025 Victor Poughon
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import torch
import colorcet as cc
import matplotlib as mpl

from typing import TypeAlias

from torchlensmaker.optical_data import OpticalData


# Color theme
color_valid = "#ffa724"
color_blocked = "red"
color_focal_point = "red"
color_spot_diagram = "coral"

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
