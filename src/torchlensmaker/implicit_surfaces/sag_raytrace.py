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

from jaxtyping import Float, Int
import torch

from .sag_functions import SagFunction2D, SagFunction3D
from .implicit_solver import (
    implicit_solver_newton,
    ImplicitFunction2D,
    ImplicitFunction3D,
    BatchTensor,
    Batch2DTensor,
    Batch3DTensor,
)


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
    sag_function: SagFunction2D,
    P: Float[torch.Tensor, "N D"],
    V: Float[torch.Tensor, "N D"],
    num_iter: Int[torch.Tensor, ""],
):
    """
    Sag surface 2D raytracing in local frame of reference
    """

    implicit_function = sag_to_implicit_2d(sag_function)
    t = implicit_solver_newton(P, V, implicit_function, num_iter)

    # Note that here the raytracing is not constrained to the domain of the sag
    # function defined by the lens diameter. The sag function typically extends
    # beyond that domain and some solution here might be outside of it.

    # Compute normals
    points = P + t.unsqueeze(-1) * V
    _, normals = implicit_function(points)

    # A surface always has two opposite normals, so keep the one pointing
    # against the ray, because that's what we need for refraction / reflection
    # i.e. the normal such that dot(normal, ray) < 0
    dot = torch.sum(normals * V, dim=-1)
    opposite_normals = torch.where(
        (dot > 0).unsqueeze(-1).expand_as(normals), -normals, normals
    )

    return t, opposite_normals
