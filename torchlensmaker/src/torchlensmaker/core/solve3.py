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


def solve3x3(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Batched solve for Ax = b, where:
        A is (..., 3, 3).
        b is (..., 3)

    matching torch.linalg.solve's broadcasting.
    Returns x with the same shape as b.
    """

    # Unpack the 9 entries. Indexing with ints keeps the leading batch dims intact.
    a00, a01, a02 = A[..., 0, 0], A[..., 0, 1], A[..., 0, 2]
    a10, a11, a12 = A[..., 1, 0], A[..., 1, 1], A[..., 1, 2]
    a20, a21, a22 = A[..., 2, 0], A[..., 2, 1], A[..., 2, 2]

    # Cofactors (the 2x2 minors, signed).
    c00 = a11 * a22 - a12 * a21
    c01 = -(a10 * a22 - a12 * a20)
    c02 = a10 * a21 - a11 * a20
    c10 = -(a01 * a22 - a02 * a21)
    c11 = a00 * a22 - a02 * a20
    c12 = -(a00 * a21 - a01 * a20)
    c20 = a01 * a12 - a02 * a11
    c21 = -(a00 * a12 - a02 * a10)
    c22 = a00 * a11 - a01 * a10

    # det(A) via expansion along the first row.
    det = a00 * c00 + a01 * c01 + a02 * c02  # shape (...)

    # Adjugate = transpose of the cofactor matrix. Stack rows of adj directly:
    # adj[i, j] = cofactor[j, i]
    inv_det = 1.0 / det
    row0 = torch.stack((c00, c10, c20), dim=-1) * inv_det.unsqueeze(-1)
    row1 = torch.stack((c01, c11, c21), dim=-1) * inv_det.unsqueeze(-1)
    row2 = torch.stack((c02, c12, c22), dim=-1) * inv_det.unsqueeze(-1)
    A_inv = torch.stack((row0, row1, row2), dim=-2)  # shape (..., 3, 3)

    # Vector RHS: (..., 3) -> treat as (..., 3, 1)
    return (A_inv @ b.unsqueeze(-1)).squeeze(-1)
