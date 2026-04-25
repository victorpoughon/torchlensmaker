import torch

from torchimplicit.math import safe_sign
from torchimplicit.types import ImplicitResult


def implicit_sphere_2d(
    points: torch.Tensor,
    R: float | torch.Tensor,
    *,
    order: int,
) -> ImplicitResult:
    """
    Implicit sphere in 2D.

    Note that this is really the projection of a 3D sphere into an abstract 2D meridional plane.
    So in practice it is a circle of radius R in the (x, r) plane.

        F(x, r) = |sqrt(x^2 + r^2) - R|

    Args:
        points : (..., 2) tensor, columns are (x, r)
        R      : sphere radius

    Returns:
        F    : (...,)      — function values
        grad : (..., 2)    — gradient
        hess : (..., 2, 2) — symmetric Hessian

    Note: singular at origin (x = r = 0)
    """
    x = points[..., 0]
    r = points[..., 1]

    # --- shared intermediates ---
    rho = (x**2 + r**2).sqrt()
    d = rho - R
    F = d.abs()
    s = safe_sign(d)
    inv_rho = 1.0 / rho

    # --- gradient ---
    grad = None
    if order >= 1:
        grad_x = s * x * inv_rho
        grad_r = s * r * inv_rho
        grad = torch.stack([grad_x, grad_r], dim=-1)

    # --- hessian ---
    hess = None
    if order >= 2:
        rho2 = rho**2
        A = s / (rho * rho2)  # s / rho^3
        H_xx = A * (rho2 - x**2)
        H_rr = A * (rho2 - r**2)
        H_xr = -A * x * r
        hess = torch.stack(
            [
                torch.stack([H_xx, H_xr], dim=-1),
                torch.stack([H_xr, H_rr], dim=-1),
            ],
            dim=-1,
        )  # (..., 2, 2)

    return ImplicitResult(F, grad, hess)


def implicit_sphere_3d(
    points: torch.Tensor,
    R: float | torch.Tensor,
    *,
    order: int,
) -> ImplicitResult:
    """
    Implicit sphere in 3D.

    The sphere is centered at the origin and has radius R.

        F(x, y, z) = |sqrt(x^2 + y^2 + z^2) - R|

    Args:
        points : (..., 3) tensor, columns are (x, y, z)
        R      : sphere radius

    Returns:
        F    : (...,)      — function values
        grad : (..., 3)    — gradient
        hess : (..., 3, 3) — symmetric Hessian

    Note: singular at origin (x = y = z = 0)
    """
    x = points[..., 0]
    y = points[..., 1]
    z = points[..., 2]

    # --- shared intermediates ---
    rho = (x**2 + y**2 + z**2).sqrt()
    d = rho - R
    F = d.abs()
    s = safe_sign(d)
    inv_rho = 1.0 / rho

    # --- gradient ---
    grad = None
    if order >= 1:
        grad_x = s * x * inv_rho
        grad_y = s * y * inv_rho
        grad_z = s * z * inv_rho
        grad = torch.stack([grad_x, grad_y, grad_z], dim=-1)

    # --- hessian ---
    hess = None
    if order >= 2:
        rho2 = rho**2
        A = s / (rho * rho2)  # s / rho^3
        H_xx = A * (rho2 - x**2)
        H_yy = A * (rho2 - y**2)
        H_zz = A * (rho2 - z**2)
        H_xy = -A * x * y
        H_xz = -A * x * z
        H_yz = -A * y * z
        hess = torch.stack(
            [
                torch.stack([H_xx, H_xy, H_xz], dim=-1),
                torch.stack([H_xy, H_yy, H_yz], dim=-1),
                torch.stack([H_xz, H_yz, H_zz], dim=-1),
            ],
            dim=-1,
        )  # (..., 3, 3)

    return ImplicitResult(F, grad, hess)
