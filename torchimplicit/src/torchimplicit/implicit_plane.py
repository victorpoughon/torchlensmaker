from multiprocessing.reduction import register

import torch

from torchimplicit.domain import total_domain
from torchimplicit.math import safe_sign
from torchimplicit.registry import example_empty, register_implicit_function
from torchimplicit.types import ImplicitFunction, ImplicitResult


def implicit_yaxis_2d(
    points: torch.Tensor,
    params: torch.Tensor,
    *,
    order: int,
) -> ImplicitResult:
    """
    Implicit function for the X=0 axis in 2D

    Params: none

    Args:
        points (Tensor): input points, shape (..., 2) where last dimension is (x, r)
        params (Tensor): tensor of parameters (empty)
        order: differentiation order

    Returns: ImplicitResult object
    """

    assert params.shape == (0,), (
        f"yaxis_2d expects parameters of shape (0,), got {params.shape}"
    )

    x = points[..., 0]
    F = torch.abs(x)
    zero = torch.zeros_like(x)
    grad = torch.stack((safe_sign(x), zero), dim=-1) if order >= 1 else None
    hess = (
        torch.zeros((*x.shape, 2, 2), dtype=x.dtype, device=x.device)
        if order >= 2
        else None
    )
    return ImplicitResult(F, grad, hess)


def implicit_yzplane_3d(
    points: torch.Tensor,
    params: torch.Tensor,
    *,
    order: int,
) -> ImplicitResult:
    """
    Implicit function for the X=0 plane in 3D

    Params: none

    Args:
        points (Tensor): input points, shape (..., 3) where last dimension is (x, y, z)
        params (Tensor): tensor of parameters (empty)
        order: differentiation order

    Returns: ImplicitResult object
    """

    assert params.shape == (0,), (
        f"yzplane_3d expects parameters of shape (0,), got {params.shape}"
    )

    x = points[..., 0]
    F = torch.abs(x)
    zero = torch.zeros_like(x)
    grad = torch.stack((safe_sign(x), zero, zero), dim=-1) if order >= 1 else None
    hess = (
        torch.zeros((*x.shape, 3, 3), dtype=x.dtype, device=x.device)
        if order >= 2
        else None
    )
    return ImplicitResult(F, grad, hess)


yaxis_2d = ImplicitFunction(
    name="yaxis_2d",
    dim=2,
    func=implicit_yaxis_2d,
    n_params=0,
    param_names=(),
    example_params=example_empty(),
    domain=total_domain,
)
register_implicit_function(yaxis_2d)


yzplane_3d = ImplicitFunction(
    name="yzplane_3d",
    dim=3,
    func=implicit_yzplane_3d,
    n_params=0,
    param_names=(),
    example_params=example_empty(),
    domain=total_domain,
)
register_implicit_function(yzplane_3d)
