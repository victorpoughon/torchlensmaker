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

from torchlensmaker.implicit.types import ImplicitResult


def implicit_yzcircle_2d(
    points: torch.Tensor,
    R: float | torch.Tensor,
    *,
    order: int = 1,
) -> ImplicitResult:
    """
    Implicit circle in 2D.

    Note that this is really the projection of a 3D circle into an abstract 2D meridional plane.
    So in practice it is actually the "surface" defined by the two points (0, R) and (0, -R).

        F(x, r) = sqrt((|r| - R)^2 + x^2)

    Args:
        points : (..., 2) tensor, columns are (x, r)
        R      : circle radius

    Returns:
        F    : (...,)      — function values
        grad : (..., 2)    — gradient
        hess : (..., 2, 2) — symmetric Hessian

    Note: singular at r = 0
    """
    x = points[..., 0]
    r = points[..., 1]

    # --- shared intermediates ---
    sgn_r = torch.sign(r)
    d = r.abs() - R  # |r| - R
    s = d**2 + x**2
    F = s.sqrt()  # (...,)
    inv_F = 1.0 / F  # 1 / sqrt(s)

    # --- gradient ---
    grad = None
    if order >= 1:
        grad_x = x * inv_F
        grad_r = sgn_r * d * inv_F
        grad = torch.stack([grad_x, grad_r], dim=-1)  # (..., 2)

    # --- hessian ---
    hess = None
    if order >= 2:
        s32 = s * F  # s^(3/2)
        A = 1.0 / s32  # 1 / s^(3/2)
        H_xx = (s - x**2) * A
        H_xr = -x * sgn_r * d * A  # = H_rx
        H_rr = (s - d**2) * A
        hess = torch.stack(
            [
                torch.stack([H_xx, H_xr], dim=-1),
                torch.stack([H_xr, H_rr], dim=-1),
            ],
            dim=-1,
        )  # (..., 2, 2)

    return ImplicitResult(F, grad, hess)


def implicit_yzcircle_3d(
    points: torch.Tensor,
    R: float | torch.Tensor,
    *,
    order: int = 1,
) -> ImplicitResult:
    """
    Implicit circle in 3D.

    The circle lies in the YZ plane, is centered at the origin and has radius R.

        F(x, y, z) = sqrt((sqrt(y^2 + z^2) - R)^2 + x^2)

    Args:
        points : (..., 3) tensor, columns are (x, y, z)
        R      : circle radius

    Returns:
        F    : (...,)      — function values
        grad : (..., 3)    — gradient
        hess : (..., 3, 3) — symmetric Hessian

    Note: singular on the x-axis (y = z = 0)
    """
    x = points[..., 0]
    y = points[..., 1]
    z = points[..., 2]

    # --- shared intermediates ---
    r = (y**2 + z**2).sqrt()
    d = r - R  # r - R
    s = d**2 + x**2
    F = s.sqrt()  # (...,)
    inv_F = 1.0 / F  # 1 / sqrt(s)
    inv_r = 1.0 / r  # 1 / r

    # --- gradient ---
    grad = None
    if order >= 1:
        grad_x = x * inv_F
        grad_y = y * d * inv_r * inv_F
        grad_z = z * d * inv_r * inv_F
        grad = torch.stack([grad_x, grad_y, grad_z], dim=-1)  # (..., 3)

    # --- hessian ---
    hess = None
    if order >= 2:
        s32 = s * F  # s^(3/2)
        r2 = r**2
        r3 = r * r2
        d2 = d**2
        A = 1.0 / s32  # 1 / s^(3/2)
        B = d * inv_r * A  # (r-R) / (r s^(3/2))
        C = (R * s - r * d2) / (r3 * s32)  # [Rs - r(r-R)²] / (r³ s^(3/2))
        diag_num = d * r2 * s  # (r-R) r² s
        diag_fac = R * s - r * d2  # R s - r (r-R)²

        H_xx = (s - x**2) * A
        H_yy = (diag_num + y**2 * diag_fac) / (r3 * s32)
        H_zz = (diag_num + z**2 * diag_fac) / (r3 * s32)
        H_xy = -x * y * B
        H_xz = -x * z * B
        H_yz = y * z * C

        hess = torch.stack(
            [
                torch.stack([H_xx, H_xy, H_xz], dim=-1),
                torch.stack([H_xy, H_yy, H_yz], dim=-1),
                torch.stack([H_xz, H_yz, H_zz], dim=-1),
            ],
            dim=-1,
        )  # (..., 3, 3)

    return ImplicitResult(F, grad, hess)
