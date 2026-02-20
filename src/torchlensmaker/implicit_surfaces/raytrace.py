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
from jaxtyping import Float, Bool
import torch

from torchlensmaker.types import (
    BatchTensor,
    BatchNDTensor,
    MaskTensor,
    Tf,
)

from torchlensmaker.kinematics.homogeneous_geometry import (
    transform_rays,
    transform_vectors,
)


# (P, V) -> t, local_normals, valid
LocalSolver: TypeAlias = Callable[
    [Float[torch.Tensor, "N D"], Float[torch.Tensor, "N D"]],
    tuple[
        Float[torch.Tensor, "  N"], Float[torch.Tensor, "N D"], Bool[torch.Tensor, " N"]
    ],
]


def raytrace(
    P: BatchNDTensor,
    V: BatchNDTensor,
    tf: Tf,
    local_solver: LocalSolver,
) -> tuple[BatchTensor, BatchNDTensor, MaskTensor]:
    """
    Surface raytracing

    Args:
        P, V: rays
        tf: kinematic transform applied to the surface
        local_solver: LocalSolver function performing raytracing in surface local frame
        domain_function: DomainFunction restricting the valid surface domain in surface local frame

    Returns:
        * t: parameter such that P + tV is the surface ray intersection point
        * normals: normal unit vectors at the intersection, such that dot(normal, V) < 0
    """

    # Convert rays to surface local frame
    P_local, V_local = transform_rays(tf.inverse, P, V)

    # Call the local solver
    t, local_normals, valid = local_solver(P_local, V_local)

    # A surface always has two opposite normals, so keep the one pointing
    # against the ray, because that's what we need for refraction / reflection
    # i.e. the normal such that dot(normal, ray) < 0
    dot = torch.sum(local_normals * V, dim=-1)
    opposite_normals = torch.where(
        (dot > 0).unsqueeze(-1).expand_as(local_normals), -local_normals, local_normals
    )

    # Convert normals to global frame
    global_normals = transform_vectors(tf.direct, opposite_normals)

    return t, global_normals, valid
