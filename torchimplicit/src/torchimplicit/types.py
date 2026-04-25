from dataclasses import dataclass, field
from typing import Protocol

import torch


@dataclass
class ImplicitResult:
    val: torch.Tensor
    grad: torch.Tensor | None = field(default=None)
    hess: torch.Tensor | None = field(default=None)


class ImplicitFunction(Protocol):
    def __call__(self, points: torch.Tensor, *, order: int) -> ImplicitResult: ...
