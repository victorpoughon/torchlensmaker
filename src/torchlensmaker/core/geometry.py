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

from torchlensmaker.core.rot2d import rot2d
from torchlensmaker.core.rot3d import euler_angles_to_matrix

from typing import TypeAlias

Tensor: TypeAlias = torch.Tensor


def unit_vector(dim: int, dtype: torch.dtype = torch.float64) -> Tensor:
    "Unit vector along the X axis"
    return torch.cat((torch.ones(1, dtype=dtype), torch.zeros(dim - 1, dtype=dtype)))


def rotated_unit_vector(angles: Tensor, dim: int) -> Tensor:
    """
    Rotated unit X vector in 2D or 3D
    angles is batched with shape (N, 2|3)
    """

    dtype = angles.dtype
    N = angles.shape[0]
    if dim == 2:
        unit = torch.tensor([1.0, 0.0], dtype=dtype)
        return rot2d(unit, angles)
    else:
        unit = torch.tensor([1.0, 0.0, 0.0], dtype=dtype)
        thetas = torch.column_stack(
            (
                torch.zeros(N, dtype=dtype),
                angles,
            )
        )
        M = euler_angles_to_matrix(thetas, "XZY").to(
            dtype=dtype
        )  # TODO need to support dtype in euler_angles_to_matrix
        return torch.matmul(M, unit.view(3, 1)).squeeze(-1)


def unit2d_rot(theta: float, dtype: torch.dtype = torch.float64) -> Tensor:
    v = torch.tensor([1.0, 0.0], dtype=dtype)
    return rot2d(v, torch.deg2rad(torch.as_tensor(theta, dtype=dtype)))


def unit3d_rot(
    theta1: float, theta2: float, dtype: torch.dtype = torch.float64
) -> Tensor:
    return rotated_unit_vector(
        torch.deg2rad(torch.as_tensor([[theta1, theta2]], dtype=dtype)), dim=3
    ).squeeze(0)


def within_radius(radius: float, points: torch.Tensor) -> torch.Tensor:
    "Mask indicating if points of shape (..., 2|3) are within 'radius' distance from the X axis"

    dim = points.shape[-1]
    if dim == 2:
        r = points.select(-1, 1)
        return torch.le(torch.abs(r), radius)
    else:
        y, z = points.select(-1, 1), points.select(-1, 2)
        return torch.le(y**2 + z**2, radius**2)


def sample_grid2d(
    N: int,
    xmin: torch.Tensor,
    xmax: torch.Tensor,
    ymin: torch.Tensor,
    ymax: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    x = torch.linspace(xmin, xmax, N, dtype=dtype)
    y = torch.linspace(ymin, ymax, N, dtype=dtype)
    X, Y = torch.meshgrid(x, y, indexing="ij")
    grid = torch.stack((X, Y), dim=-1).reshape(-1, 2)
    return grid


def sample_cylinder(
    N: int,
    xmin: torch.Tensor,
    xmax: torch.Tensor,
    tau: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Generate a grid of points within a cylinder defined by xmin, xmax, and tau.
    """
    theta = torch.linspace(0, 2 * torch.pi, N, dtype=dtype)
    r = torch.linspace(0, tau, N, dtype=dtype)
    x = torch.linspace(xmin, xmax, N, dtype=dtype)

    X, R, Theta = torch.meshgrid(x, r, theta, indexing="ij")

    Y = R * torch.cos(Theta)
    Z = R * torch.sin(Theta)
    
    grid = torch.stack((X, Y, Z), dim=-1).reshape(-1, 3)
    return grid


def sample_bcyl(N: int, xmin: torch.Tensor, xmax: torch.Tensor, tau: torch.Tensor, dim: int, dtype: torch.dtype):
    if dim == 2:
        return sample_grid2d(N, xmin, xmax, -tau, tau, dtype)
    else:
        return sample_cylinder(N, xmin, xmax, tau, dtype)
