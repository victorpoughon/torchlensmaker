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
from torchlensmaker.surfaces.implicit_solver import init_closest_origin
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


def parametric_solver_newton_step(
    theta: torch.Tensor,
    P: BatchNDTensor,
    V: BatchNDTensor,
    parametric_function: ParametricFunction,
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
    delta = solve3x3(J, Q)
    return delta


def clamp_theta(theta: torch.Tensor, clamp_positive: bool) -> torch.Tensor:
    # clamp t to positive if requested in the config
    if clamp_positive:
        theta[..., 0] = torch.clamp(theta[..., 0], min=0.0)

    # Always clamp (u,v) to (0,1)
    theta[..., 1:] = torch.clamp(theta[..., 1:], min=0.0, max=1.0)
    return theta


def parametric_solver_newton(
    P: BatchNDTensor,
    V: BatchNDTensor,
    parametric_function: ParametricFunction,
    num_iter: int,
    damping: float,
    init: float | str,
    clamp_positive: bool,
) -> tuple[BatchTensor, BatchTensor]:
    """
    First order Newton's method for parametric surfaces.
    Differentiable over the last iteration.
    """

    # Initialize θ = (t, u, v); uv always starts at center of parameter domain
    if init == "closest":
        t0 = init_closest_origin(P, V)
    else:
        t0 = torch.full_like(P[..., -1], float(init))

    # TODO better uv initialization options
    uv0 = torch.full(P.shape[:-1] + (2,), 0.5, dtype=P.dtype, device=P.device)
    theta = torch.cat([t0.unsqueeze(-1), uv0], dim=-1)

    if num_iter == 0:
        return theta[..., 0], theta[..., 1:]

    # Do N - 1 non differentiable steps
    with torch.no_grad():
        for _ in range(num_iter - 1):
            delta = parametric_solver_newton_step(theta, P, V, parametric_function)
            theta = clamp_theta(theta - damping * delta, clamp_positive)

    # One differentiable step
    delta = parametric_solver_newton_step(theta, P, V, parametric_function)
    theta = clamp_theta(theta - damping * delta, clamp_positive)

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
