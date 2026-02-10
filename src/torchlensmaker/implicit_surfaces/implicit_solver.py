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
from jaxtyping import Float
import torch

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


def implicit_solver_newton(
    P: Float[torch.Tensor, "N D"],
    V: Float[torch.Tensor, "N D"],
    implicit_function: ImplicitFunction2D | ImplicitFunction3D,
    num_iter: int,
) -> Float[torch.Tensor, " N"]:
    """
    Newton's method, diffentiable over the last iteration
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
