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

from typing import Callable, TypeAlias

import torch
from jaxtyping import Bool, Float, Int
from torch._higher_order_ops import while_loop

from torchlensmaker.implicit import ImplicitFunction
from torchlensmaker.types import (
    BatchNDTensor,
    BatchTensor,
    MaskTensor,
)

# (points) -> valid mask
DomainFunction: TypeAlias = Callable[[BatchTensor, BatchNDTensor], MaskTensor]

ImplicitSolver: TypeAlias = Callable[
    [BatchNDTensor, BatchNDTensor, ImplicitFunction], BatchTensor
]


def init_closest_origin(P: BatchNDTensor, V: BatchNDTensor) -> BatchTensor:
    return -torch.sum(P * V, dim=-1) / torch.sum(V * V, dim=-1)


def implicit_solver_newton_step(
    t: BatchTensor,
    P: BatchNDTensor,
    V: BatchNDTensor,
    implicit_function: ImplicitFunction,
) -> BatchTensor:
    "One step of first order newton solver"
    points = P + t.unsqueeze(-1) * V
    F = implicit_function(points, order=1)
    assert F.grad is not None

    delta = F.val / torch.sum(F.grad * V, dim=-1)
    return delta


def implicit_solver_newton(
    P: BatchNDTensor,
    V: BatchNDTensor,
    implicit_function: ImplicitFunction,
    num_iter: int,
    damping: float,
    init: float | str,
    clamp_positive: bool,
) -> BatchTensor:
    """
    Newton's method, diffentiable over the last iteration
    This version exports with static loop unrolling
    """

    # Initialize t
    if init == "closest":
        t = init_closest_origin(P, V)
    else:
        t = torch.full_like(P[..., -1], float(init))

    if num_iter == 0:
        return t

    # Do N - 1 non differentiable steps
    with torch.no_grad():
        for i in range(num_iter - 1):
            delta = implicit_solver_newton_step(t, P, V, implicit_function)
            t = t - damping * delta
            if clamp_positive:
                t = torch.maximum(torch.zeros_like(t), t)

    # One differentiable step
    delta = implicit_solver_newton_step(t, P, V, implicit_function)
    t = t - damping * delta
    if clamp_positive:
        t = torch.maximum(torch.zeros_like(t), t)

    return t


def implicit_solver_newton2_step(
    t: BatchTensor,
    P: BatchNDTensor,
    V: BatchNDTensor,
    implicit_function: ImplicitFunction,
) -> BatchTensor:
    "One step of second order newton solver"

    points = P + t.unsqueeze(-1) * V
    F = implicit_function(points, order=2)
    assert F.grad is not None
    assert F.hess is not None

    Q = F.val
    Qp = torch.sum(F.grad * V, dim=-1)
    R = (F.hess @ V.unsqueeze(-1)).squeeze(-1)
    Qpp = torch.sum(V * R, dim=-1)

    num = Qp * Q
    denom = Qp**2 + Qpp * Q

    delta = num / denom
    return delta


def implicit_solver_newton2(
    P: BatchNDTensor,
    V: BatchNDTensor,
    implicit_function: ImplicitFunction,
    num_iter: int,
    damping: float,
    init: float | str,
    clamp_positive: bool,
) -> BatchTensor:
    """
    Second order Newton's method, diffentiable over the last iteration
    This version exports with static loop unrolling
    Requires hessian of the implicit function
    """

    # Initialize t
    if init == "closest":
        t = init_closest_origin(P, V)
    else:
        t = torch.full_like(P[..., -1], float(init))

    if num_iter == 0:
        return t

    # Do N - 1 non differentiable steps
    with torch.no_grad():
        for _ in range(num_iter - 1):
            delta = implicit_solver_newton2_step(t, P, V, implicit_function)
            t = t - damping * delta
            if clamp_positive:
                t = torch.maximum(torch.zeros_like(t), t)

    # One differentiable step
    delta = implicit_solver_newton2_step(t, P, V, implicit_function)
    t = t - damping * delta
    if clamp_positive:
        t = torch.maximum(torch.zeros_like(t), t)

    return t


def implicit_solver_newton_while_loop(
    P: BatchNDTensor,
    V: BatchNDTensor,
    implicit_function: ImplicitFunction,
    num_iter: Int[torch.Tensor, ""],
    damping: Float[torch.Tensor, ""],
) -> BatchTensor:
    """
    Newton's method, diffentiable over the last iteration

    This version should export with ONNX Loop operator,
    when PyTorch support is ready:
    https://github.com/pytorch/pytorch/pull/162645
    https://github.com/pytorch/pytorch/issues/172568
    """

    def cond_fn(
        t: BatchTensor, i: Int[torch.Tensor, ""], n: Int[torch.Tensor, ""]
    ) -> Bool[torch.Tensor, ""]:
        return i < (n - 1)

    def body_fn(
        t: BatchTensor, i: Int[torch.Tensor, ""], n: Int[torch.Tensor, ""]
    ) -> tuple[BatchTensor, Int[torch.Tensor, ""], Int[torch.Tensor, ""]]:
        points = P + t.unsqueeze(-1) * V
        F = implicit_function(points, order=1)
        assert F.grad is not None

        delta = F.val / torch.sum(F.grad * V, dim=-1)
        t = torch.maximum(torch.zeros_like(t), t - damping * delta)
        i = i + 1
        return t, i, n.clone()

    # Initialize t at zero
    t = torch.zeros_like(P[..., -1])
    i = torch.zeros((), dtype=torch.int64, device=num_iter.device)

    # Do N - 1 non differentiable steps
    with torch.no_grad():
        t, i, _ = while_loop(cond_fn, body_fn, (t, i, num_iter))

    # One differentiable step
    t, i, _ = body_fn(t, i, num_iter)

    return t


def implicit_surface_local_raytrace(
    P: BatchNDTensor,
    V: BatchNDTensor,
    implicit_function: ImplicitFunction,
    domain_function: DomainFunction,
    implicit_solver: ImplicitSolver,
) -> tuple[BatchTensor, BatchNDTensor, MaskTensor, BatchTensor]:
    """
    Raytracing for implicit surfaces in local frame
    """

    t = implicit_solver(P, V, implicit_function)

    # Final evaluation after solver to compute normals, valid and rsm
    points = P + t.unsqueeze(-1) * V
    F = implicit_function(points, order=1)
    assert F.grad is not None

    # To get the normals of an implicit surface,
    # normalize the gradient of the implicit function
    local_normals = torch.nn.functional.normalize(F.grad, dim=-1)

    with torch.no_grad():
        # Apply the domain function to contraint to the valid domain.
        # This is required because sag functions can extend beyond the surface to be
        # modeled and also because the implicit solver can fail to converge
        valid = domain_function(F.val, P + t.unsqueeze(-1) * V)

    return t, local_normals, valid, F.val.abs()
