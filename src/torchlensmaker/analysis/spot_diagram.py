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


def get_var(data: tlm.OpticalData, name: str) -> Optional[Tensor]:
    if name == "base":
        return data.var_base
    elif name == "object":
        return data.var_object
    elif name == "wavelength":
        return data.var_wavelength
    else:
        raise RuntimeError


def spot_diagram(
    optics: nn.Module,
    sampling: dict[str, Any],
    row: Optional[str] = None,
    col: Optional[str] = None,
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

        row: string (optional)
            variable to use for the rows of the diagrams

        col: string (optional)
            variable to use for the columns of the diagram

        color_dim: string (optional)
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
    var_row = get_var(output_full, row) if row is not None else None
    var_col = get_var(output_full, col) if col is not None else None

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

    fig, axes = plt.subplots(nrows, ncols, squeeze=False, **fig_kw)

    fig.suptitle("Spot Diagram")

    # Plot image coordinate as points
    for index_row in range(nrows):
        for index_col in range(ncols):

            # Evaluate model with this spot sampling dict
            output = optics(
                tlm.default_input(
                    spot_sampling[index_row][index_col], dim=3, dtype=dtype
                )
            )

            # Get 2D image plane coordinates
            coords = output.rays_image.detach().numpy()
            assert coords.ndim == 2
            assert coords.shape[1] == 2

            # Plot image coordinate as points
            ax = axes[index_row][index_col]
            ax.scatter(coords[:, 1], coords[:, 0], s=0.5, marker=".")

    for ax in axes.flatten():
        ax.set_xlabel("Z")
        ax.set_ylabel("Y")
        ax.set_aspect("equal")
        ax.set_title("Spot Diagram")
        ax.grid()
