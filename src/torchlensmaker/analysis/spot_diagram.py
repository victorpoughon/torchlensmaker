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
import torch.nn as nn
import torchlensmaker as tlm

import matplotlib.pyplot as plt

from torchlensmaker.analysis.colors import (
    LinearSegmentedColormap,
    default_colormap,
    color_valid,
    color_rays,
    color_spot_diagram,
)

from typing import Optional, Any

Tensor = torch.Tensor


def make_var_label(name: str, value: float | list[float], precision: int):
    if isinstance(value, float):
        return f"{name}\n{value:.{precision}f}"
    elif isinstance(value, list):
        list_str = "[" + ", ".join([f"{val:.{precision}f}" for val in value]) + "]"
        return f"{name}\n{list_str}"
    else:
        return ""


def spot_diagram(
    optics: nn.Module,
    sampling: dict[str, Any],
    row: Optional[str] = None,
    col: Optional[str] = None,
    scale: bool = True,
    grid: bool = False,
    color_dim: Optional[str] = None,
    colormap: LinearSegmentedColormap = default_colormap,
    dtype: torch.dtype = torch.float64,
    **fig_kw: Any,
) -> None:
    """
    Compute and plot a spot diagram for an optical model.

    Args:
        optics: the optical model to analyse

        sampling: sampling configuration

        row: string (optional, default None)
            variable to use for the rows of the diagrams

        col: string (optional, default None)
            variable to use for the columns of the diagram

        scale: bool (optional, default True)
            Use the same scale for all spots

        grid: bool (optional, default False)
            show a grid

        **fig_kw:
            All additional keyword arguments are passed to the pyplot.figure call
    """

    # Process shortcut definitions in the sampling dict
    sampling = tlm.init_sampling(sampling)

    # Setup rows and cols, with convenience generators for iteration
    nrows = sampling[row].size() if row is not None else 1
    ncols = sampling[col].size() if col is not None else 1

    def rows():
        yield from range(nrows)

    def cols():
        yield from range(ncols)

    def spots():
        for ir in rows():
            for ic in cols():
                yield ir, ic

    # Get the "non cartesian producted" sampling variables
    # by evaluating the stack
    output_full: tlm.OpticalData = optics(
        tlm.default_input(sampling, dim=3, dtype=dtype)
    )
    var_row = output_full.get_var_optional(row) if row is not None else None
    var_col = output_full.get_var_optional(col) if col is not None else None

    # Some error checking
    if row is not None and var_row is None:
        raise RuntimeError(
            f"Requested row={repr(row)} but variable {repr(row)} is not available in the optical model"
        )
    if col is not None and var_col is None:
        raise RuntimeError(
            f"Requested col={repr(col)} but variable {repr(col)} is not available in the optical model"
        )
    if output_full.dim != 3:
        raise RuntimeError(
            f"spot diagram requires 3D optical simulation, got dim={output_full.dim}"
        )
    if output_full.rays_image is None:
        raise RuntimeError(
            "spot diagram requires image coordinates. Does your stack have an ImagePlane?"
        )

    # Build a sampling dictionary for each spot in the diagram, i.e. each row/col position
    spot_sampling: list[list[dict[str, tlm.Sampler]]] = [
        [{} for index_col in range(ncols)] for index_row in range(nrows)
    ]
    for ir, ic in spots():
        for name, sampler in sampling.items():
            if row is not None and var_row is not None and name == row:
                spot_sampling[ir][ic][name] = tlm.sampling.exact(
                    var_row[ir].unsqueeze(0)
                )
            elif col is not None and var_col is not None and name == col:
                spot_sampling[ir][ic][name] = tlm.sampling.exact(
                    var_col[ic].unsqueeze(0)
                )
            else:
                spot_sampling[ir][ic][name] = sampler

    # Compute image coordinates, and color data for each spot
    spot_coords: list[list[Tensor]] = []
    spot_colors: list[list[Tensor]] = []
    for ir in rows():
        spot_coords.append([])
        spot_colors.append([])
        for ic in cols():
            # Evaluate model with the current row/col spot sampling dict
            output = optics(
                tlm.default_input(spot_sampling[ir][ic], dim=3, dtype=dtype)
            )

            # Get 2D image plane coordinates
            coords = output.rays_image.detach()
            assert coords.ndim == 2
            assert coords.shape[1] == 2
            spot_coords[-1].append(coords)

            # Get color data
            spot_colors[-1].append(
                color_rays(output, color_dim, colormap)
                if color_dim is not None
                else color_spot_diagram
            )
    del coords
    del output

    # TODO display the number of rays per spot on the figure

    # Each tensor in spot_coords has shape [..., 2], where:
    # - the first dimension is the number of points in the spot, which beween 0
    #   (if all rays are blocked) and the product of the sizes of all non
    #   row/col dimensions
    # - the second dimension is 2 for the Y and Z axes of the image plane

    # Compute spot centers and range
    centers: list[list[Tensor]] = [
        [spot_coords[ir][ic].mean(dim=0) for ic in cols()] for ir in rows()
    ]

    centered_spots: list[list[Tensor]] = [
        [spot_coords[ir][ic] - centers[ir][ic] for ic in cols()] for ir in rows()
    ]

    # Range radius is the common display range that will be used for all spots
    range_radius = 1.1 * max(
        [torch.max(torch.abs(centered_spots[ir][ic])) for ir, ic in spots()]
    )

    fig, axes = plt.subplots(nrows, ncols, squeeze=False, **fig_kw)

    fig.suptitle("Spot Diagram")

    # Plot image coordinate as points
    for ir, ic in spots():
        coords = spot_coords[ir][ic]

        # Plot image coordinates as points
        ax = axes[ir][ic]
        ax.scatter(
            coords[:, 1],
            coords[:, 0],
            s=0.5,
            c=spot_colors[ir][ic],
            marker=".",
        )

        centerZ, centerY = centers[ir][ic]
        ax.set_xlim([centerY - range_radius, centerY + range_radius])
        ax.set_ylim([centerZ - range_radius, centerZ + range_radius])

    # Set the row / col titles
    if col is not None and var_col is not None:
        for ic, ax in enumerate(axes[0]):
            label = make_var_label(col, var_col[ic].tolist(), precision=2)
            ax.set_title(label, size="medium")

    if row is not None and var_row is not None:
        for ir, ax in zip(range(nrows), axes[:, 0]):
            label = make_var_label(row, var_row[ir].tolist(), precision=2)
            ax.set_ylabel(label, rotation=0, ha="right", size="medium")

    for ax in axes.flatten():
        ax.set_aspect("equal")
        if grid:
            ax.grid()

    fig.tight_layout()

    return fig, axes
