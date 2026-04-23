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

from typing import Any, Self

import torch
import torch.nn.functional as F

from torchlensmaker.core.functional_kernel import FunctionalKernel
from torchlensmaker.kinematics.homogeneous_geometry import (
    hom_identity_2d,
    hom_identity_3d,
)
from torchlensmaker.types import (
    BatchNDTensor,
    BatchTensor,
    MaskTensor,
    Tf,
)

from .kernels_utils import example_rays_2d, example_rays_3d
from .raytrace import surface_raytrace
from .surface_element import SurfaceElement, SurfaceElementOutput


def closest_point(
    P: BatchNDTensor,
    V: BatchNDTensor,
) -> tuple[BatchTensor, BatchNDTensor, MaskTensor, BatchTensor]:
    "Closest approach of a ray to the origin"

    t = -torch.sum(P * V, dim=-1) / torch.sum(V * V, dim=-1)
    closest = P + t.unsqueeze(-1) * V
    normals = F.normalize(closest, dim=-1)
    valid = torch.zeros_like(t, dtype=torch.bool)
    rsm = (closest * closest).sum(dim=-1)
    return t, normals, valid, rsm


class PointSurfaceKernel(FunctionalKernel):
    """
    Functional kernel for a point at the origin.

    Finds t such that the squared distance from P + tV to the origin is minimized.
    The squared distance is returned as the rsm.
    """

    inputs = {
        "P": BatchNDTensor,
        "V": BatchNDTensor,
        "tf_in": Tf,
    }

    params = {}

    outputs = {
        "t": BatchTensor,
        "normals": BatchNDTensor,
        "valid": MaskTensor,
        "points_local": BatchNDTensor,
        "points_global": BatchNDTensor,
        "rsm": BatchTensor,
    }

    def __init__(self, dim: int):
        self.dim = dim

    def apply(
        self,
        P: BatchNDTensor,
        V: BatchNDTensor,
        tf_in: Tf,
    ) -> tuple[
        BatchTensor,
        BatchNDTensor,
        MaskTensor,
        BatchNDTensor,
        BatchNDTensor,
        BatchTensor,
    ]:
        t, normals, valid, points_local, points_global, rsm = surface_raytrace(
            P, V, tf_in, closest_point
        )
        return (t, normals, valid, points_local, points_global, rsm)

    def example_inputs(
        self, dtype: torch.dtype, device: torch.device
    ) -> tuple[BatchNDTensor, BatchNDTensor, Tf]:
        if self.dim == 2:
            P, V = example_rays_2d(10, dtype, device)
            tf = hom_identity_2d(dtype, device)
        else:
            P, V = example_rays_3d(10, dtype, device)
            tf = hom_identity_3d(dtype, device)
        return P, V, tf

    def example_params(self, dtype: torch.dtype, device: torch.device) -> tuple[()]:
        return tuple()


class PointSurface(SurfaceElement):
    """
    Point surface (2D or 3D)

    Represents a single point at the origin in local coordinates.
    The rsm field of SurfaceElementOutput contains the squared distance
    from each ray to the point at its closest approach.
    """

    def __init__(self) -> None:
        super().__init__()
        self.func2d = PointSurfaceKernel(2)
        self.func3d = PointSurfaceKernel(3)

    def clone(self, **overrides: Any) -> Self:
        return type(self)(**overrides)

    def __repr__(self) -> str:
        return f"{self._get_name()}()"

    def reverse(self) -> Self:
        return self.clone()

    def forward(
        self, P: BatchNDTensor, V: BatchNDTensor, tf: Tf
    ) -> SurfaceElementOutput:
        func = self.func2d if P.shape[-1] == 2 else self.func3d
        return SurfaceElementOutput(*func.apply(P, V, tf), tf.clone(), tf.clone())

    def outer_extent(self, anchor: BatchTensor) -> BatchTensor:
        return torch.zeros_like(anchor)

    def render(self, matrix: torch.Tensor) -> None:
        return None
