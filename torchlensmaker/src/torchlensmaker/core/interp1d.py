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


def interp1d(X: torch.Tensor, Y: torch.Tensor, newX: torch.Tensor) -> torch.Tensor:
    """
    Linear interpolation of 1D data, equivalent to np.interp, but torch-native and differentiable.

    X and Y must be 1D tensors of the same length, with X strictly increasing.
    newX must lie within the closed interval [X[0], X[-1]]; values outside this
    range will trigger an assertion error.

    Args:
        X: 1D tensor of sample positions, strictly increasing.
        Y: 1D tensor of sample values, same length as X.
        newX: 1D tensor of query positions.

    Returns:
        1D tensor of interpolated values at each position in newX.
    """

    assert X.ndim == Y.ndim == 1

    # Find intervals, with special case for newX == X[0]
    indices = torch.searchsorted(X, newX)
    indices = torch.where(newX == X[0], 1, indices)

    # Subtract 1 because we want the start of the interval
    indices = indices - 1

    # Make sure all newX are within the X domain
    assert torch.min(indices) >= 0
    assert torch.max(indices) <= X.numel() - 1

    # Compute slopes
    # Careful potential div by zero here
    slopes = torch.diff(Y) / torch.diff(X)

    return Y[indices] + slopes[indices] * (newX - X[indices])
