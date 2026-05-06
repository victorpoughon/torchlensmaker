import torch

from torchimplicit.implicit_circle import (
    implicit_yzcircle_2d,
    implicit_yzcircle_3d,
)
from torchimplicit.implicit_plane import (
    implicit_yaxis_2d,
    implicit_yzplane_3d,
)
from torchimplicit.registry import register_function
from torchimplicit.types import FunctionDefinition, ImplicitResult, total_domain


def implicit_disk_2d(
    points: torch.Tensor, params: torch.Tensor, *, order: int
) -> ImplicitResult:
    """
    Implicit disk in 2D.

    Note that this is really the projection of a 3D disk into an abstract 2D meridional plane.
    So in practice it is actually the line segment defined between the two points (0, R) and (0, -R).

    Params:
        R (scalar): disk radius

    Args:
        points (Tensor): input points, shape (..., 2) where last dimension is (x, y)
        params (Tensor): tensor of parameters
        order: differentiation order

    Returns: ImplicitResult object
    """

    assert params.shape == (1,), (
        f"disk_2d expects parameters of shape (1,), got {params.shape}"
    )
    R = params[0]

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
    points: torch.Tensor, params: torch.Tensor, *, order: int
) -> ImplicitResult:
    """
    Implicit disk in 3D.

    The disk lies in the YZ plane, centered at the origin with radius R.

    Params:
        R (scalar): disk radius

    Args:
        points (Tensor): input points, shape (..., 3) where last dimension is (x, y, z)
        params (Tensor): tensor of parameters
        order: differentiation order

    Returns: ImplicitResult object containing function values, gradient and hessian.
    """

    assert params.shape == (1,), (
        f"disk_3d expects parameters of shape (1,), got {params.shape}"
    )
    R = params[0]

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


register_function(
    FunctionDefinition(
        name="disk_2d",
        dim=2,
        func=implicit_disk_2d,
        n_params=1,
        param_names=("R",),
        domain=total_domain,
    )
)


register_function(
    FunctionDefinition(
        name="disk_3d",
        dim=3,
        func=implicit_disk_3d,
        n_params=1,
        param_names=("R",),
        domain=total_domain,
    )
)
