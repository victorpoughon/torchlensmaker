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
from jaxtyping import Float, Bool
from functools import partial
import torch

from torchlensmaker.core.tensor_manip import bbroad
from .implicit_solver import ImplicitFunction2D, ImplicitFunction3D
from torchlensmaker.types import (
    Batch2DTensor,
    Batch3DTensor,
    BatchNDTensor,
    BatchTensor,
    ScalarTensor,
    MaskTensor,
    Tf,
)


from .sag_functions import (
    SagFunction2D,
    SagFunction3D,
    sag_to_implicit_2d,
    sag_to_implicit_3d,
)
from .sag_geometry import (
    lens_diameter_implicit_domain_2d,
    lens_diameter_implicit_domain_3d,
    anchor_transforms_2d,
    anchor_transforms_3d,
)
from .implicit_solver import implicit_surface_local_raytrace
from .raytrace import raytrace


def sag_surface_2d(
    sag: SagFunction2D,
    num_iter: int,
    damping: float,
    tol: float,
    P: Batch2DTensor,
    V: Batch2DTensor,
    tf_in: Tf,
    diameter: ScalarTensor,
    anchors: Float[torch.Tensor, " 2"],
    scale: ScalarTensor,
    normalize: Bool[torch.Tensor, ""],
) -> tuple[BatchTensor, Batch2DTensor, MaskTensor, Tf, Tf]:
    # Setup implicit function and domain function
    one = torch.ones((), dtype=P.dtype, device=P.device)
    tau = torch.where(normalize, diameter / 2, one)
    implicit_function = sag_to_implicit_2d(sag, tau=tau)
    domain_function = partial(
        lens_diameter_implicit_domain_2d, diameter=diameter, tol=tol
    )

    # Setup the local solver
    local_solver = partial(
        implicit_surface_local_raytrace,
        implicit_function=implicit_function,
        domain_function=domain_function,
        num_iter=num_iter,
        damping=damping,
    )

    # Compute anchor transforms from anchors and scale
    extent0, _ = sag(anchors[0] * diameter / 2)
    extent1, _ = sag(anchors[1] * diameter / 2)
    tf_surface, tf_next = anchor_transforms_2d(extent0, extent1, scale, tf_in)

    # Perform raytrace
    t, normals, valid = raytrace(P, V, tf_surface, local_solver)

    return t, normals, valid, tf_surface, tf_next


def sag_surface_3d(
    sag: SagFunction3D,
    num_iter: int,
    damping: float,
    tol: float,
    P: Batch3DTensor,
    V: Batch3DTensor,
    tf_in: Tf,
    diameter: ScalarTensor,
    anchors: Float[torch.Tensor, " 2"],
    scale: ScalarTensor,
    normalize: Bool[torch.Tensor, ""],
) -> tuple[BatchTensor, Batch3DTensor, MaskTensor, Tf, Tf]:
    # Setup implicit function and domain function
    one = torch.ones((), dtype=P.dtype, device=P.device)
    tau = torch.where(normalize, diameter / 2, one)
    implicit_function = sag_to_implicit_3d(sag, tau=tau)
    domain_function = partial(
        lens_diameter_implicit_domain_3d, diameter=diameter, tol=tol
    )

    # Setup the local solver
    local_solver = partial(
        implicit_surface_local_raytrace,
        implicit_function=implicit_function,
        domain_function=domain_function,
        num_iter=num_iter,
        damping=damping,
    )

    # Compute anchor transforms from anchors and scale
    # In 3D, take Y=tau, Z=0 as the extent point, but this is arbitrary
    extent0, _ = sag(anchors[0] * diameter / 2, torch.zeros_like(anchors[0]))
    extent1, _ = sag(anchors[1] * diameter / 2, torch.zeros_like(anchors[0]))
    tf_surface, tf_next = anchor_transforms_3d(extent0, extent1, scale, tf_in)

    # Perform raytrace
    t, normals, valid = raytrace(P, V, tf_surface, local_solver)

    return t, normals, valid, tf_surface, tf_next
