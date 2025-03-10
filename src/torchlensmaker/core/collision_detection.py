from __future__ import annotations

import torch
import math
from functools import partial

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

    def delta(
        self, surface: ImplicitSurface, P: Tensor, V: Tensor, t: Tensor, max_delta: float
    ) -> Tensor:
        raise NotImplementedError


class Newton(CollisionAlgorithm):
    "Newton's method with optional damping"

    def __init__(self, damping: float, **kwargs: Any):
        super().__init__(**kwargs)
        self.damping = damping

    def delta(
        self, surface: ImplicitSurface, P: Tensor, V: Tensor, t: Tensor, max_delta: float
    ) -> Tensor:

        F, F_grad = surface_f(surface, P, V, t)
        dot = torch.sum(F_grad * V, dim=-1)
        return self.damping * torch.where(torch.abs(dot) > torch.abs(F) / max_delta, F / dot, torch.sign(dot) * torch.sign(F) * max_delta)

    def __str__(self) -> str:
        return f"{type(self).__name__}({self.damping})"


class GD(CollisionAlgorithm):
    "Gradient Descent"

    def __init__(self, step_size: float, **kwargs: Any):
        super().__init__(**kwargs)
        self.step_size = step_size

    def delta(
        self, surface: ImplicitSurface, P: Tensor, V: Tensor, t: Tensor, max_delta: float
    ) -> Tensor:

        F, F_grad = surface_f(surface, P, V, t)
        dot = torch.sum(F_grad * V, dim=-1)
        d = 2 * self.step_size * dot * F
        return torch.clamp(d, min=-max_delta, max=max_delta)

    def __str__(self) -> str:
        return f"{type(self).__name__}({self.step_size})"


class LM(CollisionAlgorithm):
    def __init__(self, damping: float, **kwargs: Any):
        super().__init__(**kwargs)
        self.damping = damping

    def delta(
        self, surface: ImplicitSurface, P: Tensor, V: Tensor, t: Tensor, max_delta: float
    ) -> Tensor:
        "Levenberg-Marquardt"

        F, F_grad = surface_f(surface, P, V, t)
        dot = torch.sum(F_grad * V, dim=-1)
        d = torch.div(dot * F, dot**2 + self.damping)
        return torch.clamp(d, min=-max_delta, max=max_delta)

    def __str__(self) -> str:
        return f"{type(self).__name__}({self.damping})"


def init_closest_origin(surface: LocalSurface, P: Tensor, V: Tensor) -> Tensor:
    """
    Find t such that the point P+tV is:
    * in 2D, the collision of the ray with the X or Y axis, depending on 
    """
    
    return -torch.sum(P * V, dim=-1) / torch.sum(V * V, dim=-1)



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
class CollisionDetectionResult:
    # Final t values
    # Tensor of shape (N,)
    t: Tensor

    # t values used for initialization
    # Tensor of shape (B, N)
    init_t: Tensor

    # History of iteration in the coarse phase
    # Tensor of shape (B, N, HA)
    history_coarse: Optional[Tensor]

    # History of iteration in the fine phase
    # Tensor of shape (N, HB)
    history_fine: Optional[Tensor]


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

    # Algorithms
    algoA: CollisionAlgorithm
    algoB: CollisionAlgorithm
    algoC: CollisionAlgorithm
    
    close_filter_threshold: float
    num_iterA: int
    num_iterB: int

    name: str

    def __call__(
        self,
        surface: ImplicitSurface,
        P: Tensor,
        V: Tensor,
        history: bool = False,
    ) -> CollisionDetectionResult:
        
        # Sanity checks
        assert P.dim() == V.dim() == 2
        assert P.shape == V.shape
        assert P.shape[-1] in (2, 3)
        assert isinstance(P, Tensor) and P.dim() == 2
        assert isinstance(V, Tensor) and V.dim() == 2

        # Initialize solutions t
        init_t = self.init(surface, P, V)
        assert init_t.shape[1] == P.shape[0]

        # Tensor dimensions
        N = P.shape[0] # Number of rays
        D = P.shape[1] # Rays dimension (2 or 3)
        B = init_t.shape[0] # Number of solutions per ray for algo A

        # History tensors, if requested
        if history:
            history_coarse = torch.zeros((B, N, self.num_iterA), dtype=surface.dtype)
            history_fine = torch.zeros((N, self.num_iterB), dtype=surface.dtype)

        br = surface.bounding_radius()

        with torch.no_grad():
            # Iteration tensor t
            t = init_t

            # Coarse phase (multiple beams)
            for ia in range(self.num_iterA):
                t = t - self.algoA.delta(surface, P, V, t, max_delta= br / B)
                if history:
                    history_coarse[:, :, ia] = t.clone()

            # Filter step:
            # Keep the best beam, deterministically even when there are multiple collisions
            # - For each ray, identify F values below a threshold
            # - Keep the one that's closest to t=0

            # Threshold mask
            F, _ = surface_f(surface, P, V, t)
            assert F.shape == (B, N)
            close_mask = torch.abs(F) < self.close_filter_threshold
            nbclose = torch.sum(close_mask, dim=0, keepdim=True).expand((B, N))

            # Score of beams
            score = torch.where(nbclose == 0, F, torch.where(close_mask, torch.abs(t), torch.tensor(float("inf"), dtype=F.dtype)))
            _, indices = torch.min(score, dim=0)

            # Keep best beam
            assert t.shape == (B, N)
            assert indices.shape == (N,)
            t = torch.gather(t, 0, indices.unsqueeze(0))
            assert t.shape == (1, N)

            # Fine phase (single beam)
            for ib in range(self.num_iterB):
                t = t - self.algoB.delta(surface, P, V, t, max_delta=br / (B*self.num_iterA))
                if history:
                    history_fine[:, ib] = t

        # Differentiable phase: one iteration for backwards pass
        t = t - self.algoC.delta(surface, P, V, t, max_delta=br)

        return CollisionDetectionResult(
            t[0, :],
            init_t,
            history_coarse if history else None,
            history_fine if history else None
        )


default_collision_method = CollisionMethod(
    init=partial(init_brd, B=12),
    algoA=Newton(damping=0.8),
    algoB=Newton(damping=0.8),
    algoC=Newton(damping=0.8),
    num_iterA = 20,
    num_iterB=10,
    close_filter_threshold=1e-5,
    name="Default collision method"
)
