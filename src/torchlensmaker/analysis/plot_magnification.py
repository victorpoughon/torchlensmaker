# This file is part of Torch Lens Maker
# Copyright (C) 2024-present Victor Poughon
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
