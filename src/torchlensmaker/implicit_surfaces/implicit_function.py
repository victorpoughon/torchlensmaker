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


from typing import TypeAlias, Any, Sequence, Callable
from jaxtyping import Float
import torch

from torchlensmaker.core.tensor_manip import bbroad

from .sag import SagFunction2D, SagFunction3D

BatchTensor: TypeAlias = Float[torch.Tensor, "..."]
Batch2DTensor: TypeAlias = Float[torch.Tensor, "... 2"]
Batch3DTensor: TypeAlias = Float[torch.Tensor, "... 3"]
ScalarTensor: TypeAlias = Float[torch.Tensor, ""]


# x, r -> F(x, r), F_grad(x, r)
ImplicitFunction2D: TypeAlias = Callable[
    [Batch2DTensor], tuple[BatchTensor, BatchTensor]
]

# x, y, z -> F(x, y, z), F_grad(x, y, z)
ImplicitFunction3D: TypeAlias = Callable[
    [Batch3DTensor], tuple[BatchTensor, Batch3DTensor]
]


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
