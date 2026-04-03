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

from torchlensmaker.implicit.implicit_circle import (
    implicit_yzcircle_2d,
    implicit_yzcircle_3d,
)
from torchlensmaker.implicit.implicit_plane import (
    implicit_yaxis_2d,
    implicit_yzplane_3d,
)


def implicit_disk_2d(
    points: torch.Tensor, R: float | torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Implicit disk in 2D.

    Note that this is really the projection of a 3D disk into an abstract 2D meridional plane.
    So in practice it is actually the line segment defined between the two points (0, R) and (0, -R).
    """
    r = points[..., 1]
    within_cylinder = torch.abs(r) <= R

    plane_F, plane_grad, plane_hess = implicit_yaxis_2d(points)
    circle_F, circle_grad, circle_hess = implicit_yzcircle_2d(points, R)

    F = torch.where(within_cylinder, plane_F, circle_F)
    grad = torch.where(within_cylinder.unsqueeze(-1), plane_grad, circle_grad)
    hess = torch.where(
        within_cylinder.unsqueeze(-1).unsqueeze(-1), plane_hess, circle_hess
    )

    return F, grad, hess


def implicit_disk_3d(
    points: torch.Tensor, R: float | torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Implicit disk in 3D.

    The disk lies in the YZ plane, centered at the origin with radius R.

    Args:
        points : (..., 3) tensor, columns are (x, y, z)
        R      : circle radius

    Returns:
        F    : (...,)      — function values
        grad : (..., 3)    — gradient
        hess : (..., 3, 3) — symmetric Hessian
    """

    y, z = points[..., 1], points[..., 2]
    within_cylinder = y**2 + z**2 <= R**2

    plane_F, plane_grad, plane_hess = implicit_yzplane_3d(points)
    circle_F, circle_grad, circle_hess = implicit_yzcircle_3d(points, R)

    F = torch.where(within_cylinder, plane_F, circle_F)
    grad = torch.where(within_cylinder.unsqueeze(-1), plane_grad, circle_grad)
    hess = torch.where(
        within_cylinder.unsqueeze(-1).unsqueeze(-1), plane_hess, circle_hess
    )

    return F, grad, hess
