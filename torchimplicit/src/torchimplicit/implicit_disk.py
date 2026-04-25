import torch

from torchimplicit.implicit_circle import (
    implicit_yzcircle_2d,
    implicit_yzcircle_3d,
)
from torchimplicit.implicit_plane import (
    implicit_yaxis_2d,
    implicit_yzplane_3d,
)
from torchimplicit.types import ImplicitResult


def implicit_disk_2d(
    points: torch.Tensor, R: float | torch.Tensor, *, order: int
) -> ImplicitResult:
    """
    Implicit disk in 2D.

    Note that this is really the projection of a 3D disk into an abstract 2D meridional plane.
    So in practice it is actually the line segment defined between the two points (0, R) and (0, -R).
    """
    r = points[..., 1]
    within_cylinder = torch.abs(r) <= R

    planeR = implicit_yaxis_2d(points, order=order)
    circleR = implicit_yzcircle_2d(points, R, order=order)

    F = torch.where(within_cylinder, planeR.val, circleR.val)

    # --- gradient ---
    grad = None
    if order >= 1:
        assert planeR.grad is not None
        assert circleR.grad is not None
        grad = torch.where(within_cylinder.unsqueeze(-1), planeR.grad, circleR.grad)

    # --- hessian ---
    hess = None
    if order >= 2:
        assert planeR.hess is not None
        assert circleR.hess is not None
        hess = torch.where(
            within_cylinder.unsqueeze(-1).unsqueeze(-1), planeR.hess, circleR.hess
        )

    return ImplicitResult(F, grad, hess)


def implicit_disk_3d(
    points: torch.Tensor, R: float | torch.Tensor, *, order: int
) -> ImplicitResult:
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

    planeR = implicit_yzplane_3d(points, order=order)
    circleR = implicit_yzcircle_3d(points, R, order=order)

    F = torch.where(within_cylinder, planeR.val, circleR.val)

    # --- gradient ---
    grad = None
    if order >= 1:
        assert planeR.grad is not None
        assert circleR.grad is not None
        grad = torch.where(within_cylinder.unsqueeze(-1), planeR.grad, circleR.grad)

    # --- hessian ---
    hess = None
    if order >= 2:
        assert planeR.hess is not None
        assert circleR.hess is not None
        hess = torch.where(
            within_cylinder.unsqueeze(-1).unsqueeze(-1), planeR.hess, circleR.hess
        )

    return ImplicitResult(F, grad, hess)
