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
from typing import Optional

Tensor = torch.Tensor


def to_tensor(
    val: int | float | torch.Tensor | list[float | int],
    default_dtype: torch.dtype = torch.float64,
) -> Tensor:
    if isinstance(val, torch.Tensor):
        return val

    return torch.as_tensor(val, dtype=default_dtype)


def cat_optional(a: Optional[Tensor], b: Optional[Tensor]) -> Optional[Tensor]:
    "something something monad"

    if a is None and b is None:
        return None
    else:
        if a is None:
            return b
        elif b is None:
            return a
        else:
            return torch.cat((a, b), dim=0)


def filter_optional_tensor(t: Optional[Tensor], valid: Tensor) -> Optional[Tensor]:
    if t is None:
        return None

    return t[valid]


def filter_optional_mask(t: Tensor, valid: Optional[Tensor]) -> Tensor:
    if valid is None:
        return t

    return t[valid]


def cartesian_prod2d(A: Tensor, B: Tensor) -> tuple[Tensor, Tensor]:
    """
    Cartesian product of 2 batched coordinate tensors of shape (N, D) and (M, E)
    returns 2 Tensors of shape (N*M , D) and (N*M, E)
    """

    if A.dim() == 1:
        A = A.unsqueeze(1)
    if B.dim() == 1:
        B = B.unsqueeze(1)

    assert A.dim() == B.dim() == 2
    N, M = A.shape[0], B.shape[0]
    D, E = A.shape[1], B.shape[1]

    PA = torch.repeat_interleave(A, M, dim=0)
    PB = torch.tile(B, (N, 1))

    assert PA.shape == (M * N, D)
    assert PB.shape == (M * N, E)
    return PA, PB


def cartesian_prod2d_optional(
    A: Optional[Tensor], B: Optional[Tensor]
) -> tuple[Optional[Tensor], Optional[Tensor]]:
    """
    Optional version of cartesian_prod2d
    If any tensor is None, returns both tensors unmodified.
    Else, returns normal cartesian_prod2d()
    """

    if A is None or B is None:
        return A, B
    else:
        return cartesian_prod2d(A, B)


def bbroad(vector: Tensor, nbatch: int) -> Tensor:
    """
    Expoands a tensor to be compatible with the dimensions of a batched tensor
    by appending batch dimensions as needed.

    Args:
    * vector: A tensor of shape M
    * nbatch: Number of dimensions of some batched tensor

    Returns:
    * A view of the vector tensor with shape (*M, ...) that is broadcastable
      with the batched tensor.
    """
    return vector.view(*vector.shape, *([1] * nbatch))
