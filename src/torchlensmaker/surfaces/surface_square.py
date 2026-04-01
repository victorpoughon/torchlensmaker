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

import math
from functools import partial
from typing import Any, Self, cast

import torch
import torch.nn as nn

from torchlensmaker.core.functional_kernel import FunctionalKernel
from torchlensmaker.core.geometry import unit_vector
from torchlensmaker.core.tensor_manip import init_param
from torchlensmaker.kinematics.homogeneous_geometry import (
    hom_identity_2d,
    hom_identity_3d,
)
from torchlensmaker.types import (
    Batch2DTensor,
    Batch3DTensor,
    BatchNDTensor,
    BatchTensor,
    HomMatrix,
    MaskTensor,
    ScalarTensor,
    Tf,
)

from .kernels_utils import example_rays_2d, example_rays_3d
from .raytrace import surface_raytrace
from .surface_element import SurfaceElement, SurfaceElementOutput


def intersection_square_3d(
    P: Batch3DTensor,
    V: Batch3DTensor,
    side_length: ScalarTensor,
) -> tuple[BatchTensor, Batch3DTensor, MaskTensor]:
    "3D ray intersection with a square centered on the origin of the YZ plane"

    # Intersection with the YZ plane
    t = -P[..., 0] / V[..., 0]
    normals = unit_vector(3, P.dtype, P.device).expand_as(V)

    # Restrict to the square domain
    points = P + t.unsqueeze(-1) * V
    valid = (
        torch.maximum(torch.abs(points[..., 1]), torch.abs(points[..., 2]))
        <= side_length / 2
    )
    return t, normals, valid


class SquareSurfaceKernel(FunctionalKernel):
    """
    Functional kernel for a square perpendicular to the optical axis.
    Cannot be evaluated in 2D because it's not axially symmetric.
    """

    inputs = {
        "P": BatchNDTensor,
        "V": BatchNDTensor,
        "tf_in": Tf,
    }

    params = {
        "side_length": ScalarTensor,
    }

    outputs = {
        "t": BatchTensor,
        "normals": BatchNDTensor,
        "valid": MaskTensor,
        "points_local": BatchNDTensor,
        "points_global": BatchNDTensor,
    }

    def apply(
        self,
        P: BatchNDTensor,
        V: BatchNDTensor,
        tf_in: Tf,
        side_length: ScalarTensor,
    ) -> tuple[BatchTensor, BatchNDTensor, MaskTensor, BatchNDTensor, BatchNDTensor]:
        solver = intersection_square_3d
        local_solver = partial(solver, side_length=side_length)
        t, normals, valid, points_local, points_global = surface_raytrace(
            P, V, tf_in, local_solver
        )
        return t, normals, valid, points_local, points_global

    def example_inputs(
        self, dtype: torch.dtype, device: torch.device
    ) -> tuple[BatchNDTensor, BatchNDTensor, Tf]:
        P, V = example_rays_3d(10, dtype, device)
        tf = hom_identity_3d(dtype, device)
        return P, V, tf

    def example_params(
        self, dtype: torch.dtype, device: torch.device
    ) -> tuple[ScalarTensor]:
        return (torch.tensor(10.0, dtype=dtype, device=device),)


class Square(SurfaceElement):
    """
    Square surface (3D only)
    """

    def __init__(self, side_length: float | ScalarTensor):
        super().__init__()
        self.side_length = init_param(self, "side_length", side_length, False)
        self.func3d = SquareSurfaceKernel()

    def clone(self, **overrides) -> Self:
        kwargs: dict[str, Any] = dict(
            side_length=self.side_length,
        )
        return type(self)(**kwargs | overrides)

    def reverse(self) -> Self:
        return self.clone()

    def forward(self, P: BatchTensor, V: BatchTensor, tf: Tf) -> SurfaceElementOutput:
        assert P.shape[-1] == 3, (
            "Square surface can only be raytraced in 3D because it's not axially symmetric"
        )
        t, normal, valid, points_local, points_global = self.func3d.apply(
            P, V, tf, self.side_length
        )
        return SurfaceElementOutput(
            t, normal, valid, points_local, points_global, tf.clone(), tf.clone()
        )

    def outer_extent(self, anchor: ScalarTensor) -> ScalarTensor:
        return torch.zeros_like(anchor)

    def render(self) -> Any:
        max_radius = math.sqrt(2) * self.side_length / 2
        a = self.side_length.item() / 2
        clip_planes = [
            [0.0, -1.0, 0.0, a],
            [0.0, 1.0, 0.0, a],
            [0.0, 0.0, -1.0, a],
            [0.0, 0.0, 1.0, a],
        ]
        return {
            "type": "surface-plane",
            "radius": max_radius.item(),
            "clip_planes": clip_planes,
        }
