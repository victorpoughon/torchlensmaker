from __future__ import annotations

import torch
import math

from dataclasses import dataclass

Tensor = torch.Tensor

from typing import TYPE_CHECKING, Optional, Callable, Any

if TYPE_CHECKING:
    from torchlensmaker.core.surfaces import ImplicitSurface, LocalSurface


def surface_f(
    surface: ImplicitSurface, P: Tensor, V: Tensor, t: Tensor
) -> tuple[Tensor, Tensor]:
    "Compute F and F_grad for a surface with variable dimension and multiple beams per ray"

    B, N = t.shape
    D = P.shape[-1]
    
    Pbeam = P.expand((B, -1, -1))
    Vbeam = V.expand((B, -1, -1))
    points = Pbeam + t.unsqueeze(-1).expand((B, N, D)) * Vbeam

    F = surface.Fd(points)
    F_grad = surface.Fd_grad(points)

    return F, F_grad


class CollisionAlgorithm:
    """Base class for collision detection algorithms"""

    def __init__(self, max_iter: int, max_delta: Optional[float]):
        self.max_iter = max_iter
        self.max_delta = max_delta

    def delta(
        self, surface: ImplicitSurface, P: Tensor, V: Tensor, t: Tensor
    ) -> Tensor:
        raise NotImplementedError

    def clamped_delta(self, surface: ImplicitSurface, P: Tensor, V: Tensor, t: Tensor) -> Tensor:
        delta = self.delta(surface, P, V, t)
        if self.max_delta is not None:
            return torch.clamp(delta, min=-self.max_delta, max=self.max_delta)
        else:
            return delta


class Newton(CollisionAlgorithm):
    "Newton's method with optional damping"

    def __init__(self, damping: float, **kwargs: Any):
        super().__init__(**kwargs)
        self.damping = damping

    def delta(
        self, surface: ImplicitSurface, P: Tensor, V: Tensor, t: Tensor
    ) -> Tensor:

        F, F_grad = surface_f(surface, P, V, t)
        # Denominator will be zero if F_grad and V are orthogonal
        dot = torch.sum(F_grad * V, dim=-1)
        return self.damping * F / dot

    def __str__(self) -> str:
        return f"{type(self).__name__}({self.damping}, {self.max_iter})"


class GD(CollisionAlgorithm):
    "Gradient Descent"

    def __init__(self, step_size: float, **kwargs: Any):
        super().__init__(**kwargs)
        self.step_size = step_size

    def delta(
        self, surface: ImplicitSurface, P: Tensor, V: Tensor, t: Tensor
    ) -> Tensor:

        F, F_grad = surface_f(surface, P, V, t)
        dot = torch.sum(F_grad * V, dim=-1)
        return 2 * self.step_size * dot * F

    def __str__(self) -> str:
        return f"{type(self).__name__}({self.step_size}, {self.max_iter})"


class LM(CollisionAlgorithm):
    def __init__(self, damping: float, **kwargs: Any):
        super().__init__(**kwargs)
        self.damping = damping

    def delta(
        self, surface: ImplicitSurface, P: Tensor, V: Tensor, t: Tensor
    ) -> Tensor:
        "Levenberg-Marquardt"

        F, F_grad = surface_f(surface, P, V, t)
        dot = torch.sum(F_grad * V, dim=-1)
        return torch.div(dot * F, dot**2 + self.damping)

    def __str__(self) -> str:
        return f"{type(self).__name__}({self.damping}, {self.max_iter})"


def init_zeros(surface: LocalSurface, P: Tensor, V: Tensor) -> Tensor:
    "Initialize t for iterations with zeros"

    N = P.shape[0]
    return torch.zeros((N,), dtype=surface.dtype)


def init_best_axis(surface: LocalSurface, P: Tensor, V: Tensor) -> Tensor:
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


def init_brd(surface: LocalSurface, P: Tensor, V: Tensor, B: int) -> Tensor:
    """
    Bounding radius domain initialization

    Returns:
        * tensor of shape (B, N)
    """

    N, D = P.shape

    # Compute t so that P + tV is the point on the ray closest to the origin
    t = - torch.sum(P * V, dim=-1) / torch.sum(V * V, dim=-1)

    # Surface maximum bounding radius
    br = math.sqrt((surface.diameter/2)**2 + surface.extent_x()**2)

    # Sample the domain to make initialization values
    start, end = t - br, t + br
    s = torch.linspace(0, 1, B)

    # Compute start + (end - start)*s with broadcasting
    brd = start.unsqueeze(0) * (1 - s).unsqueeze(1) + end.unsqueeze(0) * s.unsqueeze(1)
    assert brd.shape == (B, N)
    return brd


@dataclass
class CollisionMethod:
    """
    Dfferentiable iterative collision detection for implicit surfaces.

    Collision detection is made up of in four phases:
    
    0. Initialize multiple starting t values for each ray by sampling each rays
       intersection with the surface bounding box.

    1. Coarse phase: run a fixed number of iterative steps of algorithm A

    2. Fine phase: pick the best t value for each ray, then run a fixed number
       of iterative steps of algorithm B

    3. Differentiable phase: Run a single step of algorithm C under pytorch autograd

    This class configures everything above (algorithms, number of steps, etc.).
    """

    # initialization method
    init: Callable[[ImplicitSurface, Tensor, Tensor], Tensor]

    # Number of solutions per ray in the coarse phase
    B: int

    # Algorithms
    algoA: CollisionAlgorithm
    algoB: CollisionAlgorithm
    algoC: CollisionAlgorithm

    name: str

    def __call__(
        self,
        surface: ImplicitSurface,
        P: Tensor,
        V: Tensor,
        history: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        
        # Sanity checks
        assert P.dim() == V.dim() == 2
        assert P.shape == V.shape
        assert P.shape[-1] in (2, 3)
        assert isinstance(P, Tensor) and P.dim() == 2
        assert isinstance(V, Tensor) and V.dim() == 2

        # Tensor dimensions
        N = P.shape[0] # Number of rays
        D = P.shape[1] # Rays dimension (2 or 3)
        H = self.algoB.num_iter if history else 1 # History (1 or num_iter of algo B)
        B = self.num_beams # Number of solutions per ray for algo A)

        # Returns init phase
        # init_t :: (N, B)

        # Returns step A
        # t_history :: (N, B, HA)

        # Returns step B
        # t_history :: (N, HB)
        
        # Returns step C
        # t_history :: (N,)

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

        # 4. Differentiable phase
        # One iteration for backwards pass
        t = t - self.algoC.clamped_delta(surface, P, V, t)

        if history:
            t_history[:, -1] = t
            return t, t_history
        else:
            return t


default_collision_method = CollisionMethod(
    init=init_best_axis,
    step0=Newton(damping=0.8, max_iter=15, max_delta=10),
)
