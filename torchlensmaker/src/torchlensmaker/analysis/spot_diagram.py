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

import itertools
from typing import Any

import matplotlib.pyplot as plt
import torch
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, LinearSegmentedColormap

from torchlensmaker.core.base_module import BaseModule
from torchlensmaker.core.ray_bundle import RayBundle
from torchlensmaker.sequential.sequential import Sequential
from torchlensmaker.sequential.sequential_data import SequentialData
from torchlensmaker.surfaces.surface_element import SurfaceElementOutput

from .CET_I2 import isoluminant_cgo_80_c38

# default colormap for spot diagrams
default_colormap = LinearSegmentedColormap.from_list("CET_I2", isoluminant_cgo_80_c38)

_COLOR_VARS = ("pupil", "field", "wavel", "source")


def _make_cell_label(sv_values: torch.Tensor) -> str:
    "Format a domain_values entry as a short axis label."
    vals = sv_values.tolist()
    if isinstance(vals, float):
        return f"{vals:.3g}"
    return "[" + ", ".join(f"{v:.3g}" for v in vals) + "]"


def spot_diagram(
    model: BaseModule,
    row: str | None = None,
    col: str | None = None,
    color: str | None = None,
    colormap: Colormap = default_colormap,
    grid: bool = False,
    dtype: torch.dtype | None = None,
    **fig_kw: Any,
) -> tuple:
    """
    Compute and plot a spot diagram for an optical model.

    Args:
        model: optical model to evaluate, the last element should be a light target
        row: ray variable to lay out as rows ("pupil", "field", "wavel", "source")
        col: ray variable to lay out as columns
        color: ray variable to use for point color
        colormap: matplotlib colormap for the color dimension
        grid: show axes grid lines
        dtype: floating-point dtype for the simulation
    """
    if not isinstance(model, Sequential):
        raise ValueError("model must be a Sequential")

    if dtype is None:
        tensor = next(itertools.chain(model.parameters(), model.buffers()), None)
        dtype = tensor.dtype if tensor is not None else torch.get_default_dtype()

    head = model[:-1]
    target = model[-1]

    # Run the head to get rays arriving at the image plane
    with torch.no_grad():
        data = head(SequentialData.empty(dim=3, dtype=dtype))
        rays = data.rays
        output = target(rays, data.fk)

    sout: SurfaceElementOutput = output.surface_outputs

    # Image plane coordinates: columns 1 and 2 of points_local (YZ in plane frame)
    # Shape (N, 2) for valid rays, (N, 2) overall (invalid rays land outside the disk)
    image_coords = sout.points_local[:, 1:].detach()  # (N, 2)
    valid = sout.valid.detach()  # (N,) bool

    # Build row / col mask lists (single all-true mask when not partitioning)
    N = rays.P.shape[0]
    all_true = torch.ones(N, dtype=torch.bool, device=rays.device)
    row_masks = rays.split_masks(row) if row is not None else [all_true]
    col_masks = rays.split_masks(col) if col is not None else [all_true]

    nrows, ncols = len(row_masks), len(col_masks)

    # Gather per-cell image coordinates (valid hits only) for range computation
    cell_coords: list[list[torch.Tensor]] = []
    for m_row in row_masks:
        row_cells = []
        for m_col in col_masks:
            m = m_row & m_col & valid
            row_cells.append(image_coords[m])
        cell_coords.append(row_cells)

    # Per-cell centers; common radius = max spread of any cell around its own center
    cell_centers: list[list[torch.Tensor]] = []
    max_spread = 0.0
    for row_cells in cell_coords:
        row_centers = []
        for coords in row_cells:
            if coords.shape[0] > 0:
                center = coords.mean(dim=0)
                spread = float((coords - center).abs().max())
            else:
                center = torch.zeros(2, dtype=image_coords.dtype)
                spread = 0.0
            row_centers.append(center)
            max_spread = max(max_spread, spread)
        cell_centers.append(row_centers)
    range_radius = max(max_spread * 1.1, 1e-9)

    # Build figure
    fig, axes = plt.subplots(nrows, ncols, squeeze=False, **fig_kw)
    fig.suptitle("Spot Diagram")

    for ir, m_row in enumerate(row_masks):
        for ic, m_col in enumerate(col_masks):
            m = m_row & m_col & valid
            cell_rays = rays.mask(m)
            coords = image_coords[m]

            ax: Axes = axes[ir][ic]
            plot_spot_diagram_cell(
                ax,
                cell_rays,
                coords,
                color=color,
                cmap=colormap,
            )

            cy, cz = cell_centers[ir][ic].tolist()
            ax.set_xlim(cz - range_radius, cz + range_radius)
            ax.set_ylim(cy - range_radius, cy + range_radius)
            ax.set_aspect("equal")
            if grid:
                ax.grid()

    # Row labels (left axis ylabel)
    if row is not None:
        sv = getattr(rays, row)
        for ir, ax in enumerate(axes[:, 0]):
            label = f"{row}\n{_make_cell_label(sv.domain_values[ir])}"
            ax.set_ylabel(label, rotation=0, ha="right", size="medium")

    # Col labels (top title)
    if col is not None:
        sv = getattr(rays, col)
        for ic, ax in enumerate(axes[0]):
            label = f"{col}\n{_make_cell_label(sv.domain_values[ic])}"
            ax.set_title(label, size="medium")

    fig.tight_layout()
    return fig, axes


def plot_spot_diagram_cell(
    ax: Axes,
    rays: RayBundle,
    coords: torch.Tensor,
    *,
    color: str | None = None,
    cmap: Colormap | str | None = None,
    limit: int | None = None,
) -> None:
    """
    Plot one cell of a spot diagram on a matplotlib Axes.

    Args:
        ax: matplotlib Axes object to draw on
        rays: RayBundle for the rays in this cell (same batch size as coords)
        coords: (N, 2) image plane coordinates (Y, Z) for each ray
        color: ray variable to color by ("pupil"/"field"/"wavel"/"source")
        cmap: colormap for the color variable
        limit: if set, use [-limit, limit] for both axes
    """
    if coords.shape[0] == 0:
        return

    y = coords[:, 0].float().numpy()
    z = coords[:, 1].float().numpy()

    if color is not None and color in _COLOR_VARS:
        sv = getattr(rays, color)
        var = sv.values
        # Reduce 2D variables (e.g. 3D pupil/field) to a scalar norm for coloring
        if var.ndim == 2:
            var = torch.linalg.vector_norm(var, dim=1)
        var = var.float().detach()
        denom = var.max() - var.min()
        c = (
            ((var - var.min()) / denom).numpy()
            if denom > 1e-4
            else torch.full_like(var, 0.5).numpy()
        )
        ax.scatter(z, y, s=0.5, c=c, cmap=cmap, marker=".", vmin=0, vmax=1)
    else:
        ax.scatter(z, y, s=0.5, c="coral", marker=".")

    if limit is not None:
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
