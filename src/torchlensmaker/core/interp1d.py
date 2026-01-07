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
    "torch version of np.interp"

    assert X.ndim == Y.ndim == 1

    # find intervals
    indices = torch.searchsorted(X, newX)

    # special case for newX == X[0]
    indices = torch.where(newX == X[0], 1, indices)

    # -1 here because we want the start of the interval
    indices = indices - 1

    # make sure all newX are within the X domain
    assert torch.min(indices) >= 0
    assert torch.max(indices) <= X.numel() - 1

    # compute slopes
    # careful potential div by zero here
    slopes = torch.diff(Y) / torch.diff(X)

    return Y[indices] + slopes[indices] * (newX - X[indices])
