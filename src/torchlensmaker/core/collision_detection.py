from __future__ import annotations

import torch

Tensor = torch.Tensor

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from torchlensmaker.core.surfaces import ImplicitSurface

def surface_f(
    surface: ImplicitSurface, P: Tensor, V: Tensor, t: Tensor
) -> tuple[Tensor, Tensor]:
    "Convinience function to call F and F_grad of an implicit surface with variable dimension"

    dim = P.shape[1]
    points = P + t.unsqueeze(1).expand_as(V) * V

    if dim == 2:
        F = surface.f(points)
        F_grad = surface.f_grad(points)
    else:
        F = surface.F(points)
        F_grad = surface.F_grad(points)

    return F, F_grad


class CollisionAlgorithm:
    """Iterative collision detection for implicit surfaces"""

    def __init__(self, num_iter: float):
        self.num_iter = num_iter
        # TODO maximum step size

    def delta(
        self, surface: ImplicitSurface, P: Tensor, V: Tensor, t: Tensor
    ) -> Tensor:
        raise NotImplementedError

    def __call__(
        self,
        surface: ImplicitSurface,
        P: Tensor,
        V: Tensor,
        init_t: Tensor,
        history: bool = False,
    ) -> Tensor:

        assert isinstance(P, Tensor) and P.dim() == 2
        assert isinstance(V, Tensor) and V.dim() == 2
        assert P.shape == V.shape
        dim = P.shape[1]
        assert dim == 2 or dim == 3

        # Initialize solutions t
        t = init_t

        if history:
            t_history = torch.empty((P.shape[0], self.num_iter + 2))
            t_history[:, 0] = init_t

        with torch.no_grad():
            for i in range(self.num_iter):
                delta = self.delta(surface, P, V, t)
                # TODO early stop if delta is small enough
                t = t - delta

                if history:
                    t_history[:, i + 1] = t

        # One iteration for backwards pass
        t = t - self.delta(surface, P, V, t)

        if history:
            t_history[:, -1] = t
            return t, t_history
        else:
            return t


def newton_delta(
    surface: ImplicitSurface,
    P: Tensor,
    V: Tensor,
    t: Tensor,
    damping: float,
) -> Tensor:
    "Compute the delta for one step of Newton's method"

    F, F_grad = surface_f(surface, P, V, t)

    # Denominator will be zero if F_grad and V are orthogonal
    dot = torch.sum(F_grad * V, dim=1)
    return damping * F / dot


class Newton(CollisionAlgorithm):
    def __init__(self, num_iter: int, damping: float):
        super().__init__(num_iter)
        self.damping = damping

    def delta(
        self, surface: ImplicitSurface, P: Tensor, V: Tensor, t: Tensor
    ) -> Tensor:
        return newton_delta(surface, P, V, t, self.damping)

    def __str__(self) -> str:
        return f"{type(self).__name__}({self.num_iter}, {self.damping})"


def gd_delta(
    surface: ImplicitSurface,
    P: Tensor,
    V: Tensor,
    t: Tensor,
    step_size: float,
) -> Tensor:
    "Compute the delta for one step of gradient descent"

    F, F_grad = surface_f(surface, P, V, t)

    dot = torch.sum(F_grad * V, dim=1)
    return 2 * step_size * dot * F


class GD(CollisionAlgorithm):
    def __init__(self, num_iter: int, step_size: float):
        super().__init__(num_iter)
        self.step_size = step_size

    def delta(
        self, surface: ImplicitSurface, P: Tensor, V: Tensor, t: Tensor
    ) -> Tensor:
        return gd_delta(surface, P, V, t, self.step_size)
    
    def __str__(self) -> str:
        return f"{type(self).__name__}({self.num_iter}, {self.step_size})"


def lm_delta(
    surface: ImplicitSurface,
    P: Tensor,
    V: Tensor,
    t: Tensor,
    damping: float,
) -> Tensor:
    "Compute the delta for one step of Levenberg-Marquardt"

    F, F_grad = surface_f(surface, P, V, t)

    dot = torch.sum(F_grad * V, dim=1)
    return dot * F / (dot**2 + damping)


class LM(CollisionAlgorithm):
    def __init__(self, num_iter: int, damping: float):
        super().__init__(num_iter)
        self.damping = damping

    def delta(
        self, surface: ImplicitSurface, P: Tensor, V: Tensor, t: Tensor
    ) -> Tensor:
        return lm_delta(surface, P, V, t, self.damping)

    def __str__(self) -> str:
        return f"{type(self).__name__}({self.num_iter}, {self.damping})"
