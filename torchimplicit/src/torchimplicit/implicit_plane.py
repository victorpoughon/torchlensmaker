import torch

from torchimplicit.math import safe_sign
from torchimplicit.types import ImplicitResult


def implicit_yaxis_2d(
    points: torch.Tensor,
    *,
    order: int,
) -> ImplicitResult:
    """
    Implicit function for the X=0 axis in 2D
    """
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
    *,
    order: int,
) -> ImplicitResult:
    """
    Implicit function for the X=0 plane in 3D
    """
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
