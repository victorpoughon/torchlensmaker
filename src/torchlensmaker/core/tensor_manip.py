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
import torch.nn as nn
from typing import Optional, Sequence, Any

Tensor = torch.Tensor


def expand_bool_tuple(n: int, t: bool | tuple[bool, ...]) -> tuple[bool, ...]:
    """
    Given a single bool or a tuple of n bools,
    returns a tuple of n bools
    """

    if isinstance(t, bool):
        return (t,) * n
    elif isinstance(t, tuple):
        if not len(t) == n:
            raise RuntimeError(
                f"Expected boolean tuple with {n} elements, got {len(t)}"
            )
        return t
    else:
        raise RuntimeError(f"Expected bool or tuple of bools, got {type(t)}")


def to_tensor(
    val: float | torch.Tensor | list[float],
    default_dtype: torch.dtype | None = None,
    default_device: torch.device | None = None,
) -> torch.Tensor:
    if isinstance(val, torch.Tensor):
        return val

    # Ensure default dtype is always floating point
    if default_dtype is None:
        default_dtype = torch.get_default_dtype()

    return torch.tensor(val, dtype=default_dtype, device=default_device)


def to_tensor_detached(
    val: float | torch.Tensor | list[float],
    default_dtype: torch.dtype | None = None,
    default_device: torch.device | None = None,
) -> torch.Tensor:
    if isinstance(val, torch.Tensor):
        return val.detach()

    # Ensure default dtype is always floating point
    if default_dtype is None:
        default_dtype = torch.get_default_dtype()

    return torch.tensor(val, dtype=default_dtype, device=default_device)


def init_param(
    parent: nn.Module,
    name: str,
    val: float | torch.Tensor | list[float],
    trainable: bool = False,
    default_dtype: torch.dtype | None = None,
    default_device: torch.device | None = None,
) -> torch.Tensor:
    """Register parameter or buffer with proper device/dtype handling."""
    t = to_tensor_detached(val, default_dtype, default_device)
    if trainable:
        p = nn.Parameter(t)
        parent.register_parameter(name, p)
        return p
    else:
        parent.register_buffer(name, t)
        return t


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


def meshgrid_flat(
    *tensors: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, ...]:
    "Like torch.meshgrid but returns flattened tensors"

    grids = torch.meshgrid(*tensors, indexing="ij")
    return tuple(g.reshape(-1) for g in grids)


def meshgrid2d_flat3(
    t1: torch.Tensor,
    t2: torch.Tensor,
    t3: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Like meshgrid but specialized for 2D inputs tensors.
    Each input tensor has shape (N_i, D_i).
    Returns three flattened tensors of shapes (M, D_i),
    where M = N1 * N2 * N3.
    """

    n1, n2, n3 = t1.shape[0], t2.shape[0], t3.shape[0]
    grids = torch.meshgrid(
        torch.arange(n1, device=t1.device),
        torch.arange(n2, device=t2.device),
        torch.arange(n3, device=t3.device),
        indexing="ij",
    )

    g1, g2, g3 = (g.reshape(-1) for g in grids)
    out1 = t1[g1]
    out2 = t2[g2]
    out3 = t3[g3]

    return out1, out2, out3


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
    Expands a tensor to be compatible with the dimensions of a batched tensor
    by appending batch dimensions as needed.

    Args:
    * vector: A tensor of shape M
    * nbatch: Number of dimensions of some batched tensor

    Returns:
    * A view of the vector tensor with shape (*M, ...) that is broadcastable
      with the batched tensor.
    """
    return vector.view(*vector.shape, *([1] * nbatch))


def is_integral(dtype: torch.dtype) -> bool:
    return dtype in {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }
