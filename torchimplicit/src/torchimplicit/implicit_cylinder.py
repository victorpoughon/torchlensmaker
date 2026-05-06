import torch

from torchimplicit.registry import register_function
from torchimplicit.types import FunctionDefinition, ImplicitResult, total_domain


def implicit_xcylinder_2d(
    points: torch.Tensor,
    params: torch.Tensor,
    *,
    order: int,
) -> ImplicitResult: ...


def implicit_xcylinder_3d(
    points: torch.Tensor,
    params: torch.Tensor,
    *,
    order: int,
) -> ImplicitResult:
    ...