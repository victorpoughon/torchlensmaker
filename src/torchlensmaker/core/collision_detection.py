# This file is part of Torch Lens Maker
# Copyright (C) 2024-present Victor Poughon
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

import torch

from dataclasses import dataclass

from typing import TYPE_CHECKING, Optional, Any

if TYPE_CHECKING:
    from torchlensmaker.surfaces.implicit_surface import ImplicitSurface
    from torchlensmaker.surfaces.local_surface import LocalSurface

Tensor = torch.Tensor


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
        self,
        surface: ImplicitSurface,
        P: Tensor,
        V: Tensor,
        t: Tensor,
        max_delta: Tensor,
    ) -> Tensor:
        raise NotImplementedError


class Newton(CollisionAlgorithm):
    "Newton's method with optional damping"

    def __init__(self, damping: float, **kwargs: Any):
        super().__init__(**kwargs)
        self.damping = damping

    def delta(
        self,
        surface: ImplicitSurface,
        P: Tensor,
        V: Tensor,
        t: Tensor,
        max_delta: Tensor,
    ) -> Tensor:
        F, F_grad = surface_f(surface, P, V, t)
        dot = torch.sum(F_grad * V, dim=-1)
        return self.damping * torch.where(
            torch.abs(dot) > torch.abs(F) / max_delta,
            F / dot,
            torch.sign(dot) * torch.sign(F) * max_delta,
        )

    def __str__(self) -> str:
        return f"{type(self).__name__}({self.damping})"


class GD(CollisionAlgorithm):
    "Gradient Descent"

    def __init__(self, step_size: float, **kwargs: Any):
        super().__init__(**kwargs)
        self.step_size = step_size

    def delta(
        self,
        surface: ImplicitSurface,
        P: Tensor,
        V: Tensor,
        t: Tensor,
        max_delta: Tensor,
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
        self,
        surface: ImplicitSurface,
        P: Tensor,
        V: Tensor,
        t: Tensor,
        max_delta: Tensor,
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

    Collision detection is made up of four phases:

    0. Initialize multiple starting t values for each ray by sampling each rays
       intersection with the surface bounding box.

    1. Coarse phase: run a fixed number of iterative steps of algorithm A,
       updating all points of all rays in parallel.

    2. Fine phase: pick the best t value for each ray, then run a fixed number
       of iterative steps of algorithm B

    3. Differentiable phase: Run a single step of algorithm C under pytorch autograd

    This class configures everything above (algorithms, number of steps, etc.).
    """

    # Algorithms
    algoA: CollisionAlgorithm
    algoB: CollisionAlgorithm
    algoC: CollisionAlgorithm

    close_filter_threshold: float
    num_iterA: int
    num_iterB: int
    B: int

    name: str

    def __call__(
        self,
        surface: ImplicitSurface,
        P: Tensor,
        V: Tensor,
        tmin: Tensor,
        tmax: Tensor,
        history: bool = False,
    ) -> CollisionDetectionResult:
        """
        tmin and tmax must be such that all points within P+tminV and P+tmaxV
        are within the valid domain of the implicit surface F function

        TODO check we are safe for any ordering of tmin, tmax, i.e. if tmax < tmin
        """

        # Sanity checks
        assert P.dim() == V.dim() == 2
        assert P.shape == V.shape
        assert P.shape[-1] in (2, 3)
        assert isinstance(P, Tensor) and P.dim() == 2
        assert isinstance(V, Tensor) and V.dim() == 2
        assert tmin.shape == tmax.shape
        assert tmin.dim() == tmax.dim() == 1
        assert tmin.shape[0] == tmax.shape[0] == P.shape[0]

        # Tensor dimensions
        (N, _), B = P.shape, self.B

        # Initialize solutions t
        t_sample = torch.linspace(0., 1., B, dtype=P.dtype)

        # init_t :: (B, N)
        t_sample.unsqueeze(-1).expand((B, N))
        init_t = tmin + t_sample.unsqueeze(-1) * (tmax - tmin)
        assert init_t.shape == (B, N)

        # History tensors, if requested
        if history:
            history_coarse = torch.zeros((B, N, self.num_iterA), dtype=surface.dtype)
            history_fine = torch.zeros((N, self.num_iterB), dtype=surface.dtype)

        max_delta = torch.abs(tmax - tmin) / B

        with torch.no_grad():
            # Iteration tensor t
            t = init_t

            # Coarse phase (multiple beams)
            for ia in range(self.num_iterA):
                t = torch.clamp(
                    t - self.algoA.delta(surface, P, V, t, max_delta),
                    min=tmin,
                    max=tmax,
                )
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
            score = torch.where(
                nbclose == 0,
                F,
                torch.where(
                    close_mask, torch.abs(t), torch.tensor(float("inf"), dtype=F.dtype)
                ),
            )
            _, indices = torch.min(score, dim=0)

            # Keep best beam
            assert t.shape == (B, N)
            assert indices.shape == (N,)
            t = torch.gather(t, 0, indices.unsqueeze(0))
            assert t.shape == (1, N)

            # Fine phase (single beam)
            for ib in range(self.num_iterB):
                t = torch.clamp(
                    t - self.algoB.delta(surface, P, V, t, max_delta),
                    min=tmin,
                    max=tmax,
                )
                if history:
                    history_fine[:, ib] = t

        # Differentiable phase: one iteration for backwards pass
        t = torch.clamp(
            t - self.algoC.delta(surface, P, V, t, max_delta=max_delta),
            min=tmin,
            max=tmax,
        )

        return CollisionDetectionResult(
            t[0, :],
            init_t,
            history_coarse if history else None,
            history_fine if history else None,
        )


default_collision_method = CollisionMethod(
    algoA=Newton(damping=0.8),
    algoB=Newton(damping=0.8),
    algoC=Newton(damping=0.8),
    num_iterA=20,
    num_iterB=10,
    B = 12,
    close_filter_threshold=1e-5,
    name="Default collision method",
)
