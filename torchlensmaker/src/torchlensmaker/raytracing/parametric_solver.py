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
        # Returns shape (order+1, order+1, UV, C)
        ...


class ParametricDomainFunction(Protocol):
    # uv: solved parameter values, shape (..., 2)
    # points: ray intersection points P + tV, shape (..., 3)
    def __call__(self, uv: BatchTensor, points: BatchNDTensor) -> MaskTensor: ...


class ParametricSolver(Protocol):
    def __call__(
        self,
        P: BatchNDTensor,
        V: BatchNDTensor,
        parametric_function: ParametricFunction,
    ) -> tuple[BatchTensor, BatchTensor]: ...


class ThetaInitFunction(Protocol):
    def __call__(
        self,
        P: BatchNDTensor,
        V: BatchNDTensor,
        parametric_function: "ParametricFunction",
    ) -> torch.Tensor:
        # Returns theta of shape (..., 3) containing initial [t, u, v]
        ...


def init_theta_closest(
    P: BatchNDTensor,
    V: BatchNDTensor,
    parametric_function: "ParametricFunction",
) -> torch.Tensor:
    "Initialize t to the closest approach to the origin, u and v to 0.5"
    t0 = init_closest_origin(P, V)
    uv0 = torch.full(P.shape[:-1] + (2,), 0.5, dtype=P.dtype, device=P.device)
    return torch.cat([t0.unsqueeze(-1), uv0], dim=-1)


def init_theta_constant(
    P: BatchNDTensor,
    V: BatchNDTensor,
    parametric_function: "ParametricFunction",
    *,
    t: float,
) -> torch.Tensor:
    "Initialize t to a constant value, u and v to 0.5"
    t0 = torch.full_like(P[..., -1], t)
    uv0 = torch.full(P.shape[:-1] + (2,), 0.5, dtype=P.dtype, device=P.device)
    return torch.cat([t0.unsqueeze(-1), uv0], dim=-1)


def init_theta_grid_search(
    P: BatchNDTensor,
    V: BatchNDTensor,
    parametric_function: "ParametricFunction",
    *,
    t_range: tuple[float, float],
    t_samples: int,
    u_range: tuple[float, float],
    u_samples: int,
    v_range: tuple[float, float],
    v_samples: int,
) -> torch.Tensor:
    "Initialize theta by grid search: pick (t, u, v) minimizing ||P + tV - S(u,v)||²"
    dtype, device = P.dtype, P.device
    batch_shape = P.shape[:-1]

    # Build 1-D grids for each parameter
    t_grid = torch.linspace(
        t_range[0], t_range[1], t_samples, dtype=dtype, device=device
    )
    u_grid = torch.linspace(
        u_range[0], u_range[1], u_samples, dtype=dtype, device=device
    )
    v_grid = torch.linspace(
        v_range[0], v_range[1], v_samples, dtype=dtype, device=device
    )

    n_uv = u_samples * v_samples

    # Build uv grid: shape (n_uv, 2)
    uu, vv = torch.meshgrid(u_grid, v_grid, indexing="ij")
    uv_grid = torch.stack([uu.reshape(-1), vv.reshape(-1)], dim=-1)

    # Evaluate surface at all uv points.
    # parametric_function requires a rank-2 input (batch, 2), so flatten the
    # ray batch and grid dims together, then reshape the result back.
    uv_flat = uv_grid.expand(batch_shape + (n_uv, 2)).reshape(-1, 2)
    S_pts = parametric_function(uv_flat, order=0)[0, 0].reshape(batch_shape + (n_uv, 3))

    # Ray points for all t values: (*batch_shape, t_samples, 3)
    # P[..., None, :] + t_grid * V[..., None, :]
    ray_pts = P.unsqueeze(-2) + t_grid.view(
        (1,) * len(batch_shape) + (t_samples, 1)
    ) * V.unsqueeze(-2)

    # Squared distances over all (t, uv) combinations: (*batch_shape, t_samples, n_uv)
    # ray_pts: (*batch_shape, t_samples, 1, 3)
    # S_pts:   (*batch_shape, 1,        n_uv, 3)
    diff = ray_pts.unsqueeze(-2) - S_pts.unsqueeze(-3)
    sq_dist = (diff * diff).sum(dim=-1)  # (*batch_shape, t_samples, n_uv)

    # Find argmin over all (t, u, v) combinations per ray, then unravel to per-dim indices
    flat_idx = sq_dist.reshape(batch_shape + (t_samples * n_uv,)).argmin(dim=-1)
    t_idx, u_idx, v_idx = torch.unravel_index(
        flat_idx, (t_samples, u_samples, v_samples)
    )

    t0 = t_grid[t_idx]
    u0 = u_grid[u_idx]
    v0 = v_grid[v_idx]

    return torch.stack([t0, u0, v0], dim=-1)


def parametric_solver_newton_step(
    theta: torch.Tensor,
    P: BatchNDTensor,
    V: BatchNDTensor,
    parametric_function: ParametricFunction,
    singular_check: bool,
) -> torch.Tensor:
    "One step of first order parametric newton solver"
    t = theta[..., 0]
    uv = theta[..., 1:]

    Sout = parametric_function(uv, order=1)
    S_val = Sout[0, 0]
    S_u = Sout[1, 0]
    S_v = Sout[0, 1]

    # Q = P + tV - S(u, v)
    Q = P + t.unsqueeze(-1) * V - S_val

    # J = [V | -S_u | -S_v], columns stacked along last dim
    J = torch.stack([V, -S_u, -S_v], dim=-1)

    # Solve J × Δθ = -Q; we return delta such that θ ← θ - delta
    delta = solve3x3(J, Q, singular_check=singular_check)
    return delta


def clamp_theta(
    theta: torch.Tensor,
    clamp_positive: bool,
    periodic_uv: tuple[bool, bool],
    u_epsilon: float,
    v_epsilon: float,
) -> torch.Tensor:
    if clamp_positive:
        clamped_t = torch.clamp(theta[..., 0], min=0.0)
    else:
        clamped_t = theta[..., 0]

    # For periodic dims, wrap with remainder so the solver can cross the periodic
    # boundary without getting pinned at the degenerate pole. For non-periodic
    # dims, clamp to [epsilon, 1 - epsilon].
    def _bound(x: torch.Tensor, periodic: bool, eps: float) -> torch.Tensor:
        return torch.remainder(x, 1.0) if periodic else torch.clamp(x, eps, 1.0 - eps)

    u = _bound(theta[..., 1], periodic_uv[0], u_epsilon)
    v = _bound(theta[..., 2], periodic_uv[1], v_epsilon)

    return torch.stack([clamped_t, u, v], dim=-1)


def parametric_solver_newton(
    P: BatchNDTensor,
    V: BatchNDTensor,
    parametric_function: ParametricFunction,
    num_iter: int,
    damping: float,
    init_fn: ThetaInitFunction,
    clamp_positive: bool,
    singular_check: bool,
    periodic_uv: tuple[bool, bool],
    u_epsilon: float,
    v_epsilon: float,
) -> tuple[BatchTensor, BatchTensor]:
    """
    First order Newton's method for parametric surfaces.
    Differentiable over the last iteration.
    """

    with torch.no_grad():
        # Initialize theta = (t, u, v) with the init method
        theta = init_fn(P, V, parametric_function)

        # Clamp the initial theta
        theta = clamp_theta(
            theta, clamp_positive, periodic_uv, u_epsilon, v_epsilon
        )

        if num_iter == 0:
            return theta[..., 0], theta[..., 1:]

        # Do N - 1 non differentiable steps
        for _ in range(num_iter - 1):
            delta = parametric_solver_newton_step(
                theta, P, V, parametric_function, singular_check
            )
            theta = clamp_theta(
                theta - damping * delta,
                clamp_positive,
                periodic_uv,
                u_epsilon,
                v_epsilon,
            )

    # One differentiable step
    delta = parametric_solver_newton_step(
        theta, P, V, parametric_function, singular_check
    )
    theta = clamp_theta(
        theta - damping * delta, clamp_positive, periodic_uv, u_epsilon, v_epsilon
    )

    return theta[..., 0], theta[..., 1:]


def parametric_solver_newton2_step(
    theta: torch.Tensor,
    P: BatchNDTensor,
    V: BatchNDTensor,
    parametric_function: ParametricFunction,
    singular_check: bool,
) -> torch.Tensor:
    "One step of second order parametric newton solver"
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
    clamp_positive: bool,
    singular_check: bool,
    periodic_uv: tuple[bool, bool],
    u_epsilon: float,
    v_epsilon: float,
) -> tuple[BatchTensor, BatchTensor]:
    """
    Second order Newton's method for parametric surfaces.
    Minimizes ||Q(θ)||² using the exact Hessian of the objective.
    Differentiable over the last iteration.
    """

    with torch.no_grad():
        # Initialize theta = (t, u, v) with the init method
        theta = init_fn(P, V, parametric_function)

        # Clamp the initial theta
        theta = clamp_theta(
            theta, clamp_positive, periodic_uv, u_epsilon, v_epsilon
        )

        if num_iter == 0:
            return theta[..., 0], theta[..., 1:]

        # Do N - 1 non differentiable steps
        for _ in range(num_iter - 1):
            delta = parametric_solver_newton2_step(
                theta, P, V, parametric_function, singular_check
            )
            theta = clamp_theta(
                theta - damping * delta,
                clamp_positive,
                periodic_uv,
                u_epsilon,
                v_epsilon,
            )

    # One differentiable step
    delta = parametric_solver_newton2_step(
        theta, P, V, parametric_function, singular_check
    )
    theta = clamp_theta(
        theta - damping * delta, clamp_positive, periodic_uv, u_epsilon, v_epsilon
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
