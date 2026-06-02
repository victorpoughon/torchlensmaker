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

import pytest
import torch

from torchlensmaker.core.solve3 import solve3x3


def test_solve3x3():
    A = torch.tensor(
        [
            [4.0, 1.0, 2.0],
            [1.0, 3.0, 0.0],
            [2.0, 0.0, 5.0],
        ],
        dtype=torch.float64,
    )
    b = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)

    x_actual = solve3x3(A, b)
    x_ref = torch.linalg.solve(A, b)

    # Tikhonov regularization introduces O(det_threshold^2 / det^2) bias;
    # for this matrix det=43, threshold=1e-3 gives ~5e-10 relative error.
    torch.testing.assert_close(x_actual, x_ref, rtol=1e-6, atol=1e-6)


def test_solve3x3_singular():
    A = torch.tensor(
        [
            [4.0, 1.0, 0.0],
            [1.0, 3.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    b = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)

    # Without the flag, Tikhonov regularization returns zeros for singular matrices
    # (det=0 => inv_det=0/(0+eps^2)=0) rather than propagating NaN.
    result = solve3x3(A, b)
    assert not torch.any(torch.isnan(result))
    assert torch.allclose(result, torch.zeros_like(result))

    # With singular_check=True, raises the same error type as torch.linalg.solve
    with pytest.raises(torch.linalg.LinAlgError):
        solve3x3(A, b, singular_check=True)


def test_solve3x3_singular_batched():
    # Batch of two matrices: one regular, one singular
    A = torch.tensor(
        [
            [[4.0, 1.0, 2.0], [1.0, 3.0, 0.0], [2.0, 0.0, 5.0]],
            [[4.0, 1.0, 0.0], [1.0, 3.0, 0.0], [2.0, 0.0, 0.0]],
        ],
        dtype=torch.float64,
    )
    b = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], dtype=torch.float64)

    with pytest.raises(torch.linalg.LinAlgError):
        solve3x3(A, b, singular_check=True)
