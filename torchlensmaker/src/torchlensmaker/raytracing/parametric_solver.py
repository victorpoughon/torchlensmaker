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


from typing import Protocol

import torch

from torchlensmaker.core.solve3 import solve3x3
from torchlensmaker.raytracing.implicit_solver import init_closest_origin
from torchlensmaker.types import (
    BatchNDTensor,
    BatchTensor,
    MaskTensor,
)


class ParametricFunction(Protocol):
    def __call__(self, uv: BatchTensor, *, order: int) -> torch.Tensor:
        # uv: (*batch, 2) — any flat batch of (u, v) parameter pairs
        # Returns (order+1, order+1, *batch, 3): surface values and partial derivatives
        ...


class ParametricDomainFunction(Protocol):
    def __call__(self, uv: BatchTensor, points: BatchNDTensor) -> MaskTensor:
        # uv: (*rays, 2) — one solved (u, v) per ray
        # points: (*rays, 3) — corresponding ray-surface intersection points P + tV
        ...


class ParametricSolver(Protocol):
    def __call__(
        self,
        P: BatchNDTensor,
        V: BatchNDTensor,
        parametric_function: ParametricFunction,
    ) -> tuple[BatchTensor, BatchTensor]:
        # P, V: (*rays, 3)
        # Returns t (*rays,) and uv (*rays, 2) — one solution per ray
        ...


class ThetaInitFunction(Protocol):
    def __call__(
        self,
        P: BatchNDTensor,
        V: BatchNDTensor,
        parametric_function: "ParametricFunction",
    ) -> torch.Tensor:
        # P, V: (*rays, 3)
        # Returns (*rays, K, 3): K candidate [t, u, v] vectors per ray.
        # K=1 for point-based inits; K=t_samples*u_samples*v_samples for grid search.
        ...


def init_theta_closest(
    P: BatchNDTensor,
    V: BatchNDTensor,
    parametric_function: "ParametricFunction",
) -> torch.Tensor:
    "Single candidate per ray (K=1): t set to closest approach to origin, u=v=0.5. Returns (*rays, 1, 3)."
    t0 = init_closest_origin(P, V)
    uv0 = torch.full(P.shape[:-1] + (2,), 0.5, dtype=P.dtype, device=P.device)
    return torch.cat([t0.unsqueeze(-1), uv0], dim=-1).unsqueeze(-2)


def init_theta_constant(
    P: BatchNDTensor,
    V: BatchNDTensor,
    parametric_function: "ParametricFunction",
    *,
    t: float,
) -> torch.Tensor:
    "Single candidate per ray (K=1): t set to a constant, u=v=0.5. Returns (*rays, 1, 3)."
    t0 = torch.full_like(P[..., -1], t)
    uv0 = torch.full(P.shape[:-1] + (2,), 0.5, dtype=P.dtype, device=P.device)
    return torch.cat([t0.unsqueeze(-1), uv0], dim=-1).unsqueeze(-2)


def init_theta_grid_search(
    P: BatchNDTensor,
    V: BatchNDTensor,
    parametric_function: "ParametricFunction",
    *,
    t_domain: tuple[float, float],
    t_samples: int,
    u_domain: tuple[float, float],
    u_samples: int,
    v_domain: tuple[float, float],
    v_samples: int,
) -> torch.Tensor:
    """Return the full Cartesian product of (t, u, v) grid values as candidates.

    Returns (*rays, K, 3) where K = t_samples * u_samples * v_samples.
    The K candidates are identical for every ray (the grid is ray-independent),
    so the ray batch dims are expanded as a view without copying data.
    Call reduce_theta_min_distance to pick the best candidate per ray.
    """
    dtype, device = P.dtype, P.device
    batch_shape = P.shape[:-1]

    t_grid = torch.linspace(t_domain[0], t_domain[1], t_samples, dtype=dtype, device=device)
    u_grid = torch.linspace(u_domain[0], u_domain[1], u_samples, dtype=dtype, device=device)
    v_grid = torch.linspace(v_domain[0], v_domain[1], v_samples, dtype=dtype, device=device)

    K = t_samples * u_samples * v_samples
    tt, uu, vv = torch.meshgrid(t_grid, u_grid, v_grid, indexing="ij")
    all_thetas = torch.stack([tt.reshape(-1), uu.reshape(-1), vv.reshape(-1)], dim=-1)  # (K, 3)

    # Broadcast (K, 3) to (*rays, K, 3) without copying
    extra_dims = (1,) * len(batch_shape)
    return all_thetas.view(extra_dims + (K, 3)).expand(batch_shape + (K, 3))


def reduce_theta_min_distance(
    thetas: torch.Tensor,
    P: BatchNDTensor,
    V: BatchNDTensor,
    parametric_function: "ParametricFunction",
) -> torch.Tensor:
    """Reduce K candidates per ray to one by picking the theta minimizing ||P + tV - S(u,v)||².

    thetas: (*rays, K, 3) — K candidate [t, u, v] vectors per ray
    P, V:   (*rays, 3)
    Returns (*rays, 3) — one [t, u, v] per ray
    """
    batch_shape = P.shape[:-1]
    K = thetas.shape[-2]

    t = thetas[..., 0]    # (*rays, K)
    uv = thetas[..., 1:]  # (*rays, K, 2)

    # Evaluate surface at all K*rays uv points, then reshape back
    S_flat = parametric_function(uv.reshape(-1, 2), order=0)[0, 0]  # (*rays * K, 3)
    S = S_flat.reshape(batch_shape + (K, 3))                          # (*rays, K, 3)

    ray_pts = P.unsqueeze(-2) + t.unsqueeze(-1) * V.unsqueeze(-2)  # (*rays, K, 3)
    sq_dist = ((ray_pts - S) ** 2).sum(dim=-1)                     # (*rays, K)

    best_idx = sq_dist.argmin(dim=-1)                                          # (*rays,)
    best_idx_expanded = best_idx.unsqueeze(-1).unsqueeze(-1).expand(*batch_shape, 1, 3)
    return thetas.gather(-2, best_idx_expanded).squeeze(-2)                    # (*rays, 3)


def parametric_solver_newton_step(
    theta: torch.Tensor,
    P: BatchNDTensor,
    V: BatchNDTensor,
    parametric_function: ParametricFunction,
    singular_check: bool,
) -> torch.Tensor:
    """One step of first order Newton solver.

    theta/P/V may have any leading batch dims (*batch, 3), e.g. (*rays, 3) for
    single-beam or (*rays, K, 3) for multi-beam with K candidates per ray.
    uv is flattened to (-1, 2) before calling parametric_function (which requires
    exactly 2D input), then outputs are reshaped back to (*batch, 3).
    """
    batch_shape = theta.shape[:-1]
    t = theta[..., 0]

    # parametric_function requires uv of shape (N, 2); flatten and reshape back
    uv_flat = theta[..., 1:].reshape(-1, 2)
    Sout = parametric_function(uv_flat, order=1)
    S_val = Sout[0, 0].reshape(batch_shape + (3,))
    S_u = Sout[1, 0].reshape(batch_shape + (3,))
    S_v = Sout[0, 1].reshape(batch_shape + (3,))

    # Q = P + tV - S(u, v)
    Q = P + t.unsqueeze(-1) * V - S_val

    # J = [V | -S_u | -S_v], columns stacked along last dim
    J = torch.stack([V, -S_u, -S_v], dim=-1)

    # Solve J × Δθ = -Q; we return delta such that θ ← θ - delta
    assert J.shape[-2:] == (3, 3)
    assert Q.shape[-1:] == (3,)
    return solve3x3(J, Q, singular_check=singular_check)


def clamp_theta(
    theta: torch.Tensor,
    t_domain: tuple[float | None, float | None],
    u_domain: tuple[float, float],
    v_domain: tuple[float, float],
    periodic_uv: tuple[bool, bool],
) -> torch.Tensor:
    # Clamp t to [lo, hi]; None on either side means unbounded in that direction
    t = torch.clamp(theta[..., 0], min=t_domain[0], max=t_domain[1])

    # For periodic dims, wrap with remainder so the solver can cross the periodic
    # boundary without getting pinned at the degenerate pole. For non-periodic
    # dims, clamp to the domain bounds.
    def _bound(x: torch.Tensor, periodic: bool, lo: float, hi: float) -> torch.Tensor:
        return torch.remainder(x, 1.0) if periodic else torch.clamp(x, lo, hi)

    u = _bound(theta[..., 1], periodic_uv[0], u_domain[0], u_domain[1])
    v = _bound(theta[..., 2], periodic_uv[1], v_domain[0], v_domain[1])

    return torch.stack([t, u, v], dim=-1)


def clamp_delta_t(
    delta: torch.Tensor,
    max_delta_t: float | None,
) -> torch.Tensor:
    "Clamp the t component of a delta step to [-max_delta_t, max_delta_t]; no-op if None."
    if max_delta_t is None:
        return delta
    return torch.cat([
        delta[..., :1].clamp(-max_delta_t, max_delta_t),
        delta[..., 1:],
    ], dim=-1)


def parametric_solver_newton(
    P: BatchNDTensor,
    V: BatchNDTensor,
    parametric_function: ParametricFunction,
    num_iter: int,
    damping: float,
    init_fn: ThetaInitFunction,
    t_domain: tuple[float | None, float | None],
    u_domain: tuple[float, float],
    v_domain: tuple[float, float],
    singular_check: bool,
    periodic_uv: tuple[bool, bool],
    max_delta_t: float | None,
) -> tuple[BatchTensor, BatchTensor]:
    """
    First order Newton's method for parametric surfaces.
    Differentiable over the last iteration.
    P, V: (*rays, 3). Returns t (*rays,) and uv (*rays, 2).
    """

    with torch.no_grad():
        # init_fn returns (*rays, K, 3); reduce picks the best candidate per ray -> (*rays, 3)
        thetas = init_fn(P, V, parametric_function)
        theta = reduce_theta_min_distance(thetas, P, V, parametric_function)

        theta = clamp_theta(theta, t_domain, u_domain, v_domain, periodic_uv)

        if num_iter == 0:
            return theta[..., 0], theta[..., 1:]

        # Do N - 1 non differentiable steps
        for _ in range(num_iter - 1):
            delta = parametric_solver_newton_step(
                theta, P, V, parametric_function, singular_check
            )
            delta = clamp_delta_t(delta, max_delta_t)
            theta = clamp_theta(
                theta - damping * delta, t_domain, u_domain, v_domain, periodic_uv
            )

    # One differentiable step
    delta = parametric_solver_newton_step(
        theta, P, V, parametric_function, singular_check
    )
    delta = clamp_delta_t(delta, max_delta_t)
    theta = clamp_theta(
        theta - damping * delta, t_domain, u_domain, v_domain, periodic_uv
    )

    return theta[..., 0], theta[..., 1:]


def parametric_solver_newton_beam(
    P: BatchNDTensor,
    V: BatchNDTensor,
    parametric_function: ParametricFunction,
    num_iter_beam: int,
    num_iter: int,
    damping: float,
    init_fn: ThetaInitFunction,
    t_domain: tuple[float | None, float | None],
    u_domain: tuple[float, float],
    v_domain: tuple[float, float],
    singular_check: bool,
    periodic_uv: tuple[bool, bool],
    max_delta_t: float | None,
) -> tuple[BatchTensor, BatchTensor]:
    """
    Multi-beam first order Newton's method for parametric surfaces.
    Runs Newton iterations over all K init candidates simultaneously before
    reducing to the best one, then refines with single-beam iterations.
    Differentiable over the last iteration.
    P, V: (*rays, 3). Returns t (*rays,) and uv (*rays, 2).

    Step 0: init_fn returns (*rays, K, 3) — all candidates, no reduction yet
    Step 1: num_iter_beam Newton iterations over all candidates (*rays, K, 3)
    Step 2: reduce to best candidate per ray -> (*rays, 3)
    Step 3: num_iter - 1 single-beam Newton iterations (non-differentiable)
    Step 4: one final differentiable Newton step
    """
    rays_shape = P.shape[:-1]  # (*rays,)

    with torch.no_grad():
        # Step 0: get all K candidates per ray, shape (*rays, K, 3)
        thetas = init_fn(P, V, parametric_function)
        K = thetas.shape[-2]
        thetas = clamp_theta(thetas, t_domain, u_domain, v_domain, periodic_uv)

        # Step 1: multi-beam Newton iterations over (*rays, K, 3)
        # Expand P, V from (*rays, 3) to (*rays, K, 3) to match thetas
        P_beam = P.unsqueeze(-2).expand(rays_shape + (K, 3))
        V_beam = V.unsqueeze(-2).expand(rays_shape + (K, 3))
        for _ in range(num_iter_beam):
            delta = parametric_solver_newton_step(
                thetas, P_beam, V_beam, parametric_function, singular_check
            )
            delta = clamp_delta_t(delta, max_delta_t)
            thetas = clamp_theta(
                thetas - damping * delta, t_domain, u_domain, v_domain, periodic_uv
            )

        # Step 2: reduce to one candidate per ray, shape (*rays, 3)
        theta = reduce_theta_min_distance(thetas, P, V, parametric_function)
        theta = clamp_theta(theta, t_domain, u_domain, v_domain, periodic_uv)

        if num_iter == 0:
            return theta[..., 0], theta[..., 1:]

        # Step 3: num_iter - 1 non-differentiable single-beam refinement steps
        for _ in range(num_iter - 1):
            delta = parametric_solver_newton_step(
                theta, P, V, parametric_function, singular_check
            )
            delta = clamp_delta_t(delta, max_delta_t)
            theta = clamp_theta(
                theta - damping * delta, t_domain, u_domain, v_domain, periodic_uv
            )

    # Step 4: one differentiable step
    delta = parametric_solver_newton_step(
        theta, P, V, parametric_function, singular_check
    )
    delta = clamp_delta_t(delta, max_delta_t)
    theta = clamp_theta(
        theta - damping * delta, t_domain, u_domain, v_domain, periodic_uv
    )

    return theta[..., 0], theta[..., 1:]


def parametric_solver_newton2_step(
    theta: torch.Tensor,
    P: BatchNDTensor,
    V: BatchNDTensor,
    parametric_function: ParametricFunction,
    singular_check: bool,
) -> torch.Tensor:
    "One step of second order Newton solver. theta/P/V are (*rays, 3) — one value per ray."
    t = theta[..., 0]
    uv = theta[..., 1:]

    Sout = parametric_function(uv, order=2)
    S_val = Sout[0, 0]
    S_u = Sout[1, 0]
    S_v = Sout[0, 1]
    S_uu = Sout[2, 0]
    S_uv = Sout[1, 1]
    S_vv = Sout[0, 2]

    # Q = P + tV - S(u, v)
    Q = P + t.unsqueeze(-1) * V - S_val

    # J = [V | -S_u | -S_v], columns stacked along last dim, shape (..., 3, 3)
    J = torch.stack([V, -S_u, -S_v], dim=-1)

    # Gradient: ∇D/2 = J^T Q, shape (..., 3)
    grad = (J.mT @ Q.unsqueeze(-1)).squeeze(-1)

    # Hessian/2 = J^T J - C
    JtJ = J.mT @ J

    # Second-derivative correction: Q dotted with each second partial of S
    Quu = torch.sum(Q * S_uu, dim=-1)
    Quv = torch.sum(Q * S_uv, dim=-1)
    Qvv = torch.sum(Q * S_vv, dim=-1)

    zeros = torch.zeros_like(Quu)
    C = torch.stack(
        [
            torch.stack([zeros, zeros, zeros], dim=-1),
            torch.stack([zeros, Quu, Quv], dim=-1),
            torch.stack([zeros, Quv, Qvv], dim=-1),
        ],
        dim=-2,
    )

    H_half = JtJ - C

    # Solve (H/2) Δθ = -∇D/2; return delta such that θ ← θ - delta
    delta = solve3x3(H_half, grad, singular_check=singular_check)
    return delta


def parametric_solver_newton2(
    P: BatchNDTensor,
    V: BatchNDTensor,
    parametric_function: ParametricFunction,
    num_iter: int,
    damping: float,
    init_fn: ThetaInitFunction,
    t_domain: tuple[float | None, float | None],
    u_domain: tuple[float, float],
    v_domain: tuple[float, float],
    singular_check: bool,
    periodic_uv: tuple[bool, bool],
    max_delta_t: float | None,
) -> tuple[BatchTensor, BatchTensor]:
    """
    Second order Newton's method for parametric surfaces.
    Minimizes ||Q(θ)||² using the exact Hessian of the objective.
    Differentiable over the last iteration.
    P, V: (*rays, 3). Returns t (*rays,) and uv (*rays, 2).
    """

    with torch.no_grad():
        # init_fn returns (*rays, K, 3); reduce picks the best candidate per ray -> (*rays, 3)
        thetas = init_fn(P, V, parametric_function)
        theta = reduce_theta_min_distance(thetas, P, V, parametric_function)

        theta = clamp_theta(theta, t_domain, u_domain, v_domain, periodic_uv)

        if num_iter == 0:
            return theta[..., 0], theta[..., 1:]

        # Do N - 1 non differentiable steps
        for _ in range(num_iter - 1):
            delta = parametric_solver_newton2_step(
                theta, P, V, parametric_function, singular_check
            )
            delta = clamp_delta_t(delta, max_delta_t)
            theta = clamp_theta(
                theta - damping * delta, t_domain, u_domain, v_domain, periodic_uv
            )

    # One differentiable step
    delta = parametric_solver_newton2_step(
        theta, P, V, parametric_function, singular_check
    )
    delta = clamp_delta_t(delta, max_delta_t)
    theta = clamp_theta(
        theta - damping * delta, t_domain, u_domain, v_domain, periodic_uv
    )

    return theta[..., 0], theta[..., 1:]


def parametric_residual_domain(
    uv: BatchTensor,
    points: BatchNDTensor,
    parametric_function: ParametricFunction,
    tol: float,
) -> MaskTensor:
    "Domain function based on residual ||P + tV - S(uv)|| < tol"
    S_val = parametric_function(uv, order=1)[0, 0]
    return torch.linalg.norm(points - S_val, dim=-1) < tol


def parametric_surface_local_raytrace(
    P: BatchNDTensor,
    V: BatchNDTensor,
    parametric_function: ParametricFunction,
    solver: ParametricSolver,
    domain_function: ParametricDomainFunction,
) -> tuple[BatchTensor, BatchNDTensor, MaskTensor, BatchTensor]:
    """
    Raytracing for parametric surfaces in local frame
    """

    t, uv = solver(P, V, parametric_function)

    # Final evaluation after solver to compute normals, valid and rsm
    Sout = parametric_function(uv, order=1)
    S_val = Sout[0, 0]
    S_u = Sout[1, 0]
    S_v = Sout[0, 1]

    # To get the normals of a parametric surface,
    # cross product the two parametric partial derivatives
    cnormals = torch.linalg.cross(S_u, S_v)
    local_normals = torch.nn.functional.normalize(cnormals, dim=-1)

    points = P + t.unsqueeze(-1) * V
    rsm = torch.linalg.norm(points - S_val, dim=-1)

    with torch.no_grad():
        valid = domain_function(uv, points)

    return t, local_normals, valid, rsm
