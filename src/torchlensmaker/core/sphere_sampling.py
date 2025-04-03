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
from typing import TypeAlias

Tensor: TypeAlias = torch.Tensor


def sphere_samples_angular(
    radius: int | float | Tensor,
    start: int | float | Tensor,
    end: int | float | Tensor,
    N: int,
    dtype: torch.dtype,
) -> Tensor:
    "Angular sampling of a circular arc defined by radius"
    R, start, end = map(torch.as_tensor, (radius, start, end))

    if R > 0:
        theta = torch.linspace(torch.pi - end, torch.pi - start, N, dtype=dtype)
    else:
        theta = torch.linspace(start, end, N, dtype=dtype)

    X = torch.abs(R) * torch.cos(theta) + R
    Y = torch.abs(R) * torch.sin(theta)

    return torch.stack((X, Y), dim=-1)


def sphere_samples_linear(
    curvature: int | float | Tensor,
    start: int | float | Tensor,
    end: int | float | Tensor,
    N: int,
    dtype: torch.dtype,
) -> Tensor:
    "Linear sampling of a circular arc defined by curvature"

    curvature, start, end = map(torch.as_tensor, (curvature, start, end))

    Y = torch.linspace(start, end, N, dtype=dtype)
    Y2 = Y**2
    C = curvature

    X = torch.div(C * Y2, 1 + torch.sqrt(1 - Y2 * C**2))
    return torch.stack((X, Y), dim=-1)
