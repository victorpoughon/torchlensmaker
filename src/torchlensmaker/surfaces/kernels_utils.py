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

from torchlensmaker.types import BatchNDTensor


def example_rays_2d(
    N: int, dtype: torch.dtype, device: torch.device
) -> tuple[BatchNDTensor, BatchNDTensor]:
    P = torch.stack(
        (
            torch.zeros((N,), dtype=dtype, device=device),
            torch.linspace(-1, 1, N, dtype=dtype, device=device),
        ),
        dim=-1,
    )

    V = torch.tensor([[1.0, 0.0]], dtype=dtype, device=device).expand_as(P)

    return P, V


def example_rays_3d(
    N: int, dtype: torch.dtype, device: torch.device
) -> tuple[BatchNDTensor, BatchNDTensor]:
    P = torch.stack(
        (
            torch.zeros((N,), dtype=dtype, device=device),
            torch.linspace(-1, 1, N, dtype=dtype, device=device),
            torch.linspace(-1, 1, N, dtype=dtype, device=device),
        ),
        dim=-1,
    )

    V = torch.tensor([[1.0, 0.0, 0.0]], dtype=dtype, device=device).expand_as(P)

    return P, V
