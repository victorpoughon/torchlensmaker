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
import numpy as np

from torchlensmaker.core.interp1d import interp1d


def test_interp1d_matches_numpy():
    X = torch.linspace(0.0, 10.0, 11)
    Y = X ** 2
    newX = torch.tensor([0.0, 0.5, 2.5, 7.3, 10.0])

    result = interp1d(X, Y, newX)
    expected = torch.tensor(np.interp(newX.numpy(), X.numpy(), Y.numpy()), dtype=torch.float32)

    torch.testing.assert_close(result, expected)


def test_interp1d_at_knots():
    X = torch.tensor([0.0, 1.0, 2.0, 3.0])
    Y = torch.tensor([1.0, 4.0, 2.0, 5.0])

    result = interp1d(X, Y, X)

    torch.testing.assert_close(result, Y)


def test_interp1d_linear_segment():
    X = torch.tensor([0.0, 1.0])
    Y = torch.tensor([0.0, 2.0])
    newX = torch.tensor([0.25, 0.5, 0.75])

    result = interp1d(X, Y, newX)
    expected = torch.tensor([0.5, 1.0, 1.5])

    torch.testing.assert_close(result, expected)
