from dataclasses import dataclass, field
from typing import Protocol

import torch


@dataclass(frozen=True)
class ImplicitResult:
    """
    Result object with tensors for functions values, gradients and hessians:
        F    : (...,)      — function values
        grad : (..., D)    — gradient
        hess : (..., D, D) — symmetric Hessian
    """

    val: torch.Tensor
    grad: torch.Tensor | None = field(default=None)
    hess: torch.Tensor | None = field(default=None)


class EvalImplicitFunction(Protocol):
    def __call__(
        self,
        points: torch.Tensor,
        params: torch.Tensor,
        *,
        order: int,
    ) -> ImplicitResult: ...


class BoundImplicitFunction(Protocol):
    def __call__(
        self,
        points: torch.Tensor,
        *,
        order: int,
    ) -> ImplicitResult: ...


class DomainFunction(Protocol):
    "Domain function for implicit functions"

    def __call__(
        self,
        points: torch.Tensor,
        params: torch.Tensor,
    ) -> torch.Tensor: ...


class ExampleParamsFunction(Protocol):
    "Domain function for implicit functions"

    def __call__(
        self,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor: ...


def total_domain(points: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    return torch.full(points.shape[:-1], True, dtype=torch.bool, device=points.device)


@dataclass(frozen=True)
class ImplicitFunction:
    name: str
    dim: int  # 2 or 3
    func: EvalImplicitFunction
    n_params: int
    param_names: tuple[str, ...]
    example_params: ExampleParamsFunction
    domain: DomainFunction

    def __post_init__(self):
        if len(self.param_names) != self.n_params:
            raise ValueError(
                f"param_names has {len(self.param_names)} entries "
                f"but n_params is {self.n_params}"
            )

    def __call__(
        self, points: torch.Tensor, params: torch.Tensor, *, order: int
    ) -> ImplicitResult:
        return self.func(points, params, order=order)
