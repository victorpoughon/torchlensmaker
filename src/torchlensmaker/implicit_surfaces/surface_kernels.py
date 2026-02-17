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

from functools import partial
from typing import Any

from typing import TypeAlias
from jaxtyping import Float, Int, Bool
import torch


from torchlensmaker.types import (
    ScalarTensor,
    BatchTensor,
    Batch2DTensor,
    Batch3DTensor,
    BatchNDTensor,
    MaskTensor,
    HomMatrix,
    Tf2D,
    Tf3D,
)

from torchlensmaker.kinematics.homogeneous_geometry import (
    hom_scale_2d,
    hom_identity_2d,
    hom_translate_2d,
    kinematic_chain_append,
    kinematic_chain_extend_2d,
)


from torchlensmaker.core.functional_kernel import FunctionalKernel
from .sag_functions import spherical_sag_2d, SagFunction2D

from .sag_raytrace import sag_surface_local_raytrace_2d, raytrace


def example_rays_2d(
    N: int, dtype: torch.dtype, device: torch.device
) -> tuple[BatchNDTensor, BatchNDTensor]:
    P = torch.stack(
        (
            torch.zeros((N,), dtype=dtype, device=device),
            torch.linspace(-1, 1, N, dtype=dtype, device=device),
        ),
        dim=-1,
    )

    V = torch.tensor([[1.0, 0.0]], dtype=dtype, device=device).expand_as(P)

    return P, V


def lens_diameter_domain_2d(
    points: Batch2DTensor, diameter: ScalarTensor
) -> MaskTensor:
    return torch.abs(points[..., 1]) <= diameter / 2


def sag_anchor_transforms_2d(
    sag: SagFunction2D,
    diameter: ScalarTensor,
    anchors: Float[torch.Tensor, " 2"],
    scale: ScalarTensor,
    base: Tf2D,
) -> tuple[Tf2D, Tf2D]:
    # First anchor transform
    extent0, _ = sag(anchors[0] * diameter / 2)
    t0_x = -scale * extent0
    t0_y = torch.zeros_like(t0_x)
    tf0 = hom_translate_2d(torch.stack((t0_x, t0_y), dim=-1))

    # Second anchor transform
    extent1, _ = sag(anchors[1] * diameter / 2)
    t1_x = scale * extent1
    t1_y = torch.zeros_like(t0_y)
    tf1 = hom_translate_2d(torch.stack((t1_x, t1_y), dim=-1))

    # Compose with the existing kinematic chain
    tfscale = hom_scale_2d(scale)
    tf_surface = kinematic_chain_extend_2d(base, [tf0, tfscale])
    tf_next = kinematic_chain_extend_2d(base, [tf0, tf1])

    return tf_surface, tf_next


class SphereC2DSurfaceKernel(FunctionalKernel):
    """
    Functional kernel for a 2D spherical arc parameterized by:
        - signed surface curvature
        - lens diameter

    with support for anchors and scale.
    """

    inputs = {
        "P": Batch2DTensor,
        "V": Batch2DTensor,
        "tf_in": Tf2D,
    }

    params = {
        "diameter": ScalarTensor,
        "C": ScalarTensor,
        "anchors": Float[torch.Tensor, " 2"],
        "scale": ScalarTensor,
    }

    outputs = {
        "t": BatchTensor,
        "normals": Batch2DTensor,
        "valid": MaskTensor,
        "surface_tf": Tf2D,
        "next_tf": Tf2D,
    }

    @staticmethod
    def forward(
        P: Batch2DTensor,
        V: Batch2DTensor,
        tf_in: Tf2D,
        diameter: ScalarTensor,
        C: ScalarTensor,
        anchors: Float[torch.Tensor, " 2"],
        scale: ScalarTensor,
    ) -> tuple[BatchTensor, Batch2DTensor, MaskTensor, Tf2D, Tf2D]:
        # TODO static kernel parameter?
        num_iter: int = 3

        # Setup the local solver for this surface class
        sag = partial(spherical_sag_2d, C=C)
        local_solver = partial(
            sag_surface_local_raytrace_2d,
            sag_function=sag,
            num_iter=num_iter,
        )

        # Compute anchor transforms from anchors and scale
        tf_surface, tf_next = sag_anchor_transforms_2d(
            sag, diameter, anchors, scale, tf_in
        )

        # Domain function defined by the lens diamter
        domain_function = partial(lens_diameter_domain_2d, diameter=diameter)

        # Perform raytrace
        t, normals, valid = raytrace(P, V, tf_surface, local_solver, domain_function)

        return t, normals, valid, tf_surface, tf_next

    @staticmethod
    def example_inputs(
        dtype: torch.dtype, device: torch.device
    ) -> tuple[Batch2DTensor, Batch2DTensor, Tf2D]:
        P, V = example_rays_2d(10, dtype, device)
        tf = hom_identity_2d(dtype, device)
        return P, V, tf

    @staticmethod
    def example_params(
        dtype: torch.dtype, device: torch.device
    ) -> tuple[ScalarTensor, ScalarTensor, Float[torch.Tensor, " 2"], ScalarTensor]:
        return (
            torch.tensor(10.0, dtype=dtype, device=device),
            torch.tensor(0.5, dtype=dtype, device=device),
            torch.tensor((0.0, 0.0), dtype=dtype, device=device),
            torch.tensor(-1.0, dtype=dtype, device=device),
        )
