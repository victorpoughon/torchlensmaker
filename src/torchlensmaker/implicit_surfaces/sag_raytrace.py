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

from jaxtyping import Float
from typing import TypeAlias, Callable
import torch

from torchlensmaker.types import (
    BatchTensor,
    Batch2DTensor,
    Batch3DTensor,
    BatchNDTensor,
    MaskTensor,
)

from .sag_functions import SagFunction2D, SagFunction3D
from .implicit_solver import (
    implicit_solver_newton,
    ImplicitFunction2D,
    ImplicitFunction3D,
)

# (points) -> valid mask
DomainFunction: TypeAlias = Callable[[BatchNDTensor], MaskTensor]


def sag_to_implicit_2d(sag: SagFunction2D) -> ImplicitFunction2D:
    "Wrap a 2D sag function into an implicit function"

    def implicit(points: Batch2DTensor) -> tuple[BatchTensor, BatchTensor]:
        x, r = points.unbind(-1)
        g, g_grad = sag(r)
        f = g - x
        f_grad = torch.stack((-torch.ones_like(x), g_grad), dim=-1)
        return f, f_grad

    return implicit


def sag_to_implicit_3d(sag: SagFunction3D) -> ImplicitFunction3D:
    "Wrap a 3D sag function into an implicit function"

    def implicit(points: Batch3DTensor) -> tuple[BatchTensor, Batch3DTensor]:
        x, y, z = points.unbind(-1)
        g, g_grad = sag(y, z)
        f = g - x
        grad_x = -torch.ones_like(x)
        grad_y, grad_z = g_grad.unbind(-1)
        f_grad = torch.stack((grad_x, grad_y, grad_z), dim=-1)

        return f, f_grad

    return implicit


def sag_surface_local_raytrace_2d(
    P: Float[torch.Tensor, "N D"],
    V: Float[torch.Tensor, "N D"],
    sag_function: SagFunction2D,
    domain_function: DomainFunction,
    num_iter: int,
) -> tuple[BatchTensor, Batch2DTensor]:
    """
    Raytracing for a sag surface in 2D in local frame
    """

    implicit_function = sag_to_implicit_2d(sag_function)
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
