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

from typing import TypeAlias, Callable
from jaxtyping import Float, Int, Bool
import torch

from torchlensmaker.types import (
    BatchTensor,
    Batch2DTensor,
    Batch3DTensor,
    BatchNDTensor,
    MaskTensor,
)

from torch._higher_order_ops import while_loop

# x, r -> F(x, r), F_grad(x, r)
ImplicitFunction2D: TypeAlias = Callable[
    [Batch2DTensor], tuple[BatchTensor, Batch2DTensor]
]

# x, y, z -> F(x, y, z), F_grad(x, y, z)
ImplicitFunction3D: TypeAlias = Callable[
    [Batch3DTensor], tuple[BatchTensor, Batch3DTensor]
]

# (points) -> valid mask
DomainFunction: TypeAlias = Callable[[BatchNDTensor], MaskTensor]


def implicit_solver_newton(
    P: Float[torch.Tensor, "N D"],
    V: Float[torch.Tensor, "N D"],
    implicit_function: ImplicitFunction2D | ImplicitFunction3D,
    num_iter: int,
) -> Float[torch.Tensor, " N"]:
    """
    Newton's method, diffentiable over the last iteration
    This version exports with static loop unrolling
    """

    # Initialize t at zero
    t = torch.zeros_like(P[..., -1])

    if num_iter == 0:
        return t

    # Do N - 1 non differentiable steps
    with torch.no_grad():
        for i in range(num_iter - 1):
            points = P + t.unsqueeze(-1) * V
            F, F_grad = implicit_function(points)

            delta = F / torch.sum(F_grad * V, dim=-1)
            t = t - delta

    # One differentiable step
    points = P + t.unsqueeze(-1) * V
    F, F_grad = implicit_function(points)
    delta = F / torch.sum(F_grad * V, dim=-1)
    t = t - delta

    return t


def implicit_solver_newton_while_loop(
    P: Float[torch.Tensor, "N D"],
    V: Float[torch.Tensor, "N D"],
    implicit_function: ImplicitFunction2D | ImplicitFunction3D,
    num_iter: Int[torch.Tensor, ""],
) -> Float[torch.Tensor, " N"]:
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
        F, F_grad = implicit_function(points)

        delta = F / torch.sum(F_grad * V, dim=-1)
        t = t - delta
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
    P: Float[torch.Tensor, "N D"],
    V: Float[torch.Tensor, "N D"],
    implicit_function: ImplicitFunction2D | ImplicitFunction3D,
    domain_function: DomainFunction,
    num_iter: int,
) -> tuple[BatchTensor, Batch2DTensor, MaskTensor]:
    """
    Raytracing for a sag surface in 2D in local frame
    """

    t = implicit_solver_newton(P, V, implicit_function, num_iter)

    # To get the normals of an implicit surface,
    # normalize the gradient of the implicit function
    points = P + t.unsqueeze(-1) * V
    _, Fgrad = implicit_function(points)
    local_normals = torch.nn.functional.normalize(Fgrad, dim=-1)

    # Apply the domain function to contraint to the valid domain.
    # This is required because typically sag functions extend beyond
    # that domain or even to infinity
    valid = domain_function(P + t.unsqueeze(-1) * V)

    return t, local_normals, valid
