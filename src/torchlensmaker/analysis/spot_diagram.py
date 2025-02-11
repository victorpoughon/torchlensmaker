import torch
import torch.nn as nn
import torchlensmaker as tlm

import matplotlib.pyplot as plt

from torchlensmaker.analysis.colors import (
    LinearSegmentedColormap,
    default_colormap,
)

from typing import Optional, Any

Tensor = torch.Tensor

def make_var_label(name: str, value: float | list[float], precision: int):
    if isinstance(value, float):
        return f"{name}={value:.{precision}f}"
    elif isinstance(value, list):
        list_str = "[" + ", ".join([f"{val:.{precision}f}" for val in value]) + "]"
        return f"{name}={list_str}"
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

        color_dim: string (optional, default None)
            Dimension to use for coloring the points

        **fig_kw:
            All additional keyword arguments are passed to the pyplot.figure call
    """

    # Process shortcut definitions in the sampling dict
    sampling = tlm.init_sampling(sampling)

    # Split the input sampling configuration in one per row / col
    nrows = sampling[row].size() if row is not None else 1
    ncols = sampling[col].size() if col is not None else 1
    spot_sampling: list[list[dict[str, tlm.Sampler]]] = [
        [{} for index_col in range(ncols)] for index_row in range(nrows)
    ]

    # Get the "non cartesian producted" sampling variables
    # by evaluating the stack
    output_full = optics(tlm.default_input(sampling, dim=3, dtype=dtype))
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
    for index_row in range(nrows):
        for index_col in range(ncols):
            for name, sampler in sampling.items():
                if row is not None and var_row is not None and name == row:
                    spot_sampling[index_row][index_col][name] = tlm.sampling.exact(
                        var_row[index_row].unsqueeze(0)
                    )
                elif col is not None and var_col is not None and name == col:
                    spot_sampling[index_row][index_col][name] = tlm.sampling.exact(
                        var_col[index_col].unsqueeze(0)
                    )
                else:
                    spot_sampling[index_row][index_col][name] = sampler

    # Compute image coordinates for each spot
    spot_coords: list[list[Tensor]] = []
    for index_row in range(nrows):
        spot_coords.append([])
        for index_col in range(ncols):
            # Evaluate model with the current row/col spot sampling dict
            output = optics(
                tlm.default_input(
                    spot_sampling[index_row][index_col], dim=3, dtype=dtype
                )
            )
            
            # Get 2D image plane coordinates
            coords = output.rays_image.detach()
            assert coords.ndim == 2
            assert coords.shape[1] == 2
            spot_coords[-1].append(coords)
    del coords
    del output
    
    # spot_tensor :: [nrows, ncols, N, 2]
    # N is the product of the sizes of all non row/col dimensions
    # The last dimensions is 2 for the Y and Z axes of the image plane
    spot_tensor = torch.stack([torch.stack(row, dim=0) for row in spot_coords], dim=0)
    assert spot_tensor.shape[0] == nrows
    assert spot_tensor.shape[1] == ncols
    assert spot_tensor.shape[3] == 2

    # Compute spot centers and range
    centers = spot_tensor.mean(dim=2)
    centered_spots = spot_tensor - centers.unsqueeze(2)
    range_radius = 1.1*torch.max(torch.abs(centered_spots))

    fig, axes = plt.subplots(nrows, ncols, squeeze=False, **fig_kw)

    fig.suptitle("Spot Diagram")

    # Plot image coordinate as points
    for index_row in range(nrows):
        for index_col in range(ncols):
            coords = spot_tensor[index_row, index_col, :, :]

            # Plot image coordinates as points
            ax = axes[index_row][index_col]
            ax.scatter(coords[:, 1], coords[:, 0], s=0.5, marker=".")

            centerY = centers[index_row, index_col, 1]
            centerZ = centers[index_row, index_col, 0]
            ax.set_xlim([centerY - range_radius, centerY + range_radius])
            ax.set_ylim([centerZ - range_radius, centerZ + range_radius])

    # Set the row / col titles
    if col is not None and var_col is not None:
        for index_col, ax in enumerate(axes[0]):
            label = make_var_label(col, var_col[index_col].tolist(), precision=2)
            ax.set_title(label, size="medium")
    
    if row is not None and var_row is not None:
        for index_row, ax in zip(range(nrows), axes[:, 0]):
            label = make_var_label(row, var_row[index_row].tolist(), precision=2)
            ax.set_ylabel(label, rotation=0, ha="right", size="medium")

    for ax in axes.flatten():
        ax.set_aspect("equal")
        if grid:
            ax.grid()

    fig.tight_layout()
    
    return fig, axes
