from __future__ import annotations

import torch

from dataclasses import dataclass

Tensor = torch.Tensor

from typing import TYPE_CHECKING, Optional, Callable

if TYPE_CHECKING:
    from torchlensmaker.core.surfaces import ImplicitSurface


def surface_f(
    surface: ImplicitSurface, P: Tensor, V: Tensor, t: Tensor
) -> tuple[Tensor, Tensor]:
    "Convinience function to call F and F_grad of an implicit surface with variable dimension"

    points = P + t.unsqueeze(1).expand_as(V) * V

    F = surface.Fd(points)
    F_grad = surface.Fd_grad(points)

    return F, F_grad


class CollisionAlgorithm:
    """Base class for collision detection algorithms"""

    def __init__(self, max_iter: float, max_delta: Optional[float]):
        self.max_iter = max_iter
        self.max_delta = max_delta

    def delta(
        self, surface: ImplicitSurface, P: Tensor, V: Tensor, t: Tensor
    ) -> Tensor:
        raise NotImplementedError

    def clamped_delta(self, surface, P, V, t) -> Tensor:
        delta = self.delta(surface, P, V, t)
        if self.max_delta is not None:
            return torch.clamp(delta, min=-self.max_delta, max=self.max_delta)
        else:
            return delta


class Newton(CollisionAlgorithm):
    "Newton's method with optional damping"

    def __init__(self, damping: float, **kwargs):
        super().__init__(**kwargs)
        self.damping = damping

    def delta(
        self, surface: ImplicitSurface, P: Tensor, V: Tensor, t: Tensor
    ) -> Tensor:

        F, F_grad = surface_f(surface, P, V, t)
        # Denominator will be zero if F_grad and V are orthogonal
        dot = torch.sum(F_grad * V, dim=1)
        return self.damping * F / dot

    def __str__(self) -> str:
        return f"{type(self).__name__}({self.damping}, {self.max_iter})"


class GD(CollisionAlgorithm):
    "Gradient Descent"

    def __init__(self, step_size: float, **kwargs):
        super().__init__(**kwargs)
        self.step_size = step_size

    def delta(
        self, surface: ImplicitSurface, P: Tensor, V: Tensor, t: Tensor
    ) -> Tensor:

        F, F_grad = surface_f(surface, P, V, t)
        dot = torch.sum(F_grad * V, dim=1)
        return 2 * self.step_size * dot * F

    def __str__(self) -> str:
        return f"{type(self).__name__}({self.step_size}, {self.max_iter})"


class LM(CollisionAlgorithm):
    def __init__(self, damping: float, **kwargs):
        super().__init__(**kwargs)
        self.damping = damping

    def delta(
        self, surface: ImplicitSurface, P: Tensor, V: Tensor, t: Tensor
    ) -> Tensor:
        "Levenberg-Marquardt"

        F, F_grad = surface_f(surface, P, V, t)
        dot = torch.sum(F_grad * V, dim=1)
        return dot * F / (dot**2 + self.damping)

    def __str__(self) -> str:
        return f"{type(self).__name__}({self.damping}, {self.max_iter})"


def init_zeros(surface: ImplicitSurface, P: Tensor, V: Tensor) -> Tensor:
    "Initialize t for iterations with zeros"

    N = P.shape[0]
    return torch.zeros((N,), dtype=surface.dtype)


def init_best_axis(surface: ImplicitSurface, P: Tensor, V: Tensor) -> Tensor:
    """
    Initialize t for iterations with the rays intersections with either X or Y
    axis, depending on which is most perpendicular to the ray.
    """

    # Initial guess is the intersection of rays with the X=0 or Y=O plane,
    # depending on if rays are mostly vertical or mostly horizontal
    # TODO make this backwards safe with an inner where()
    init_x = -P[:, 0] / V[:, 0]
    init_y = -P[:, 1] / V[:, 1]
    init_t = torch.where(torch.abs(V[:, 0]) > torch.abs(V[:, 1]), init_x, init_y)

    return init_t


@dataclass
class CollisionMethod:
    """
    Dfferentiable iterative collision detection for implicit surfaces.

    A collision "method" is made up of:
    * An initialization function
    * A collision algorithm
    """

    # initialization method
    init: Callable[[ImplicitSurface, Tensor, Tensor], Tensor]

    # algorithm
    step0: CollisionAlgorithm

    def __str__(self) -> str:
        return f"{self.step0}[{self.init.__name__}]"

    def __call__(
        self,
        surface: ImplicitSurface,
        P: Tensor,
        V: Tensor,
        history: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:

        # Sanity checks
        dim = P.shape[1]
        assert dim == 2 or dim == 3
        assert isinstance(P, Tensor) and P.dim() == 2
        assert isinstance(V, Tensor) and V.dim() == 2
        assert P.shape == V.shape

        # Initialize solutions t
        t = self.init(surface, P, V)

        if history:
            t_history = torch.empty((P.shape[0], self.step0.max_iter + 2))
            t_history[:, 0] = t

        with torch.no_grad():
            for i in range(self.step0.max_iter):
                delta = self.step0.clamped_delta(surface, P, V, t)
                t = t - delta

                if history:
                    t_history[:, i + 1] = t

        # One iteration for backwards pass
        t = t - self.step0.clamped_delta(surface, P, V, t)

        if history:
            t_history[:, -1] = t
            return t, t_history
        else:
            return t


default_collision_method = CollisionMethod(
    init=init_zeros,
    step0=Newton(damping=0.8, max_iter=15, max_delta=10),
)
