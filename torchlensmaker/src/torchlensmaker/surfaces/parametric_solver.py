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


from typing import Any

import torch

from torchlensmaker.surfaces.implicit_solver import init_closest_origin
from torchlensmaker.types import (
    BatchNDTensor,
    BatchTensor,
    MaskTensor,
)

# TODO introduce Protocol based type for parametric_function


def parametric_solver_newton_step(
    theta: torch.Tensor,
    P: BatchNDTensor,
    V: BatchNDTensor,
    parametric_function: Any,
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
    delta = torch.linalg.solve(J, Q.unsqueeze(-1)).squeeze(-1)
    return delta


def parametric_solver_newton(
    P: BatchNDTensor,
    V: BatchNDTensor,
    parametric_function: Any,
    num_iter: int,
    damping: float,
) -> tuple[BatchTensor, BatchTensor]:
    """
    First order Newton's method for parametric surfaces.
    Differentiable over the last iteration.
    """

    # Initialize θ = (t, u, v)
    t0 = init_closest_origin(P, V)
    uv0 = torch.full(P.shape[:-1] + (2,), 0.5, dtype=P.dtype, device=P.device)
    theta = torch.cat([t0.unsqueeze(-1), uv0], dim=-1)

    if num_iter == 0:
        return theta[..., 0], theta[..., 1:]

    # Do N - 1 non differentiable steps
    with torch.no_grad():
        for _ in range(num_iter - 1):
            delta = parametric_solver_newton_step(theta, P, V, parametric_function)
            theta = theta - damping * delta

    # One differentiable step
    delta = parametric_solver_newton_step(theta, P, V, parametric_function)
    theta = theta - damping * delta

    return theta[..., 0], theta[..., 1:]


def parametric_solver(
    P: BatchNDTensor, V: BatchNDTensor, parametric_function: Any
) -> tuple[torch.Tensor, torch.Tensor]:
    # TODO use SolverConfig — hardcoded for now
    return parametric_solver_newton(P, V, parametric_function, num_iter=5, damping=1.0)


def parametric_surface_local_raytrace(
    P: BatchNDTensor,
    V: BatchNDTensor,
    parametric_function: Any,
) -> tuple[BatchTensor, BatchNDTensor, MaskTensor, BatchTensor]:
    """
    Raytracing for parametric surfaces in local frame
    """

    t, uv = parametric_solver(P, V, parametric_function)

    # Final evaluation after solver to compute normals, valid and rsm
    Sout = parametric_function(uv, order=1)
    S_u = Sout[1, 0]
    S_v = Sout[0, 1]

    # To get the normals of a parametric surface,
    # cross product the two parametric partial derivatives
    cnormals = torch.linalg.cross(S_u, S_v)
    local_normals = torch.nn.functional.normalize(cnormals, dim=-1)

    # TODO domain function equivalent here
    # TODO compute rsm from Q = P + tV - S(uv)

    # valid: for now, return all true while we are testing the solver
    rsm = torch.zeros_like(t)
    valid = torch.ones_like(t, dtype=torch.bool)

    return t, local_normals, valid, rsm
