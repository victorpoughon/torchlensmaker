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
from typing import Any, Self

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
from .sag_geometry import lens_diameter_domain_2d, lens_diameter_domain_3d
from .surface_element import SurfaceElement, SurfaceElementOutput

import tlmviewer as tlmv


def intersection_plane_2d(
    P: Batch2DTensor,
    V: Batch2DTensor,
) -> tuple[BatchTensor, Batch2DTensor, MaskTensor, BatchTensor]:
    "2D ray intersection with the Y axis"

    t = -P[..., 0] / V[..., 0]
    normals = unit_vector(2, P.dtype, P.device).expand_as(V)

    zero = torch.zeros((), dtype=V.dtype, device=V.device)
    valid = torch.logical_not(torch.isclose(V[..., 0], zero))
    rsm = torch.where(valid, torch.zeros_like(t), P[:, 0])
    return t, normals, valid, rsm


def intersection_plane_3d(
    P: Batch2DTensor,
    V: Batch2DTensor,
) -> tuple[BatchTensor, Batch2DTensor, MaskTensor, BatchTensor]:
    "2D ray intersection with the Y axis"

    t = -P[..., 0] / V[..., 0]
    normals = unit_vector(3, P.dtype, P.device).expand_as(V)

    zero = torch.zeros((), dtype=V.dtype, device=V.device)
    valid = torch.logical_not(torch.isclose(V[..., 0], zero))
    rsm = torch.where(valid, torch.zeros_like(t), P[:, 0])
    return t, normals, valid, rsm


class PlaneSurfaceKernel(FunctionalKernel):
    """
    Functional kernel for an infinite plane perpendicular to the principal axis
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
        local_solver = intersection_plane_2d if self.dim == 2 else intersection_plane_3d
        t, normals, valid, points_local, points_global, rsm = surface_raytrace(
            P, V, tf_in, local_solver
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


class Plane(SurfaceElement):
    """
    Plane surface (2D or 3D)

    A plane has infinite area.
    """

    def __init__(self, display_diameter: float | ScalarTensor):
        super().__init__()
        self.display_diameter = init_param(
            self, "display_diameter", display_diameter, False
        )
        self.func2d = PlaneSurfaceKernel(2)
        self.func3d = PlaneSurfaceKernel(3)

    def clone(self, **overrides: Any) -> Self:
        kwargs: dict[str, Any] = dict(display_diameter=self.display_diameter)
        return type(self)(**kwargs | overrides)

    def __repr__(self) -> str:
        return f"{self._get_name()}(display_diameter={self.display_diameter.item()})"

    def reverse(self) -> Self:
        return self.clone()

    def forward(
        self, P: BatchNDTensor, V: BatchNDTensor, tf: Tf
    ) -> SurfaceElementOutput:
        func = self.func2d if P.shape[-1] == 2 else self.func3d
        return SurfaceElementOutput(*func.apply(P, V, tf), tf.clone(), tf.clone())

    def outer_extent(self, anchor: ScalarTensor) -> ScalarTensor:
        return torch.zeros_like(anchor)

    def render(self, matrix: torch.Tensor) -> tlmv.SurfaceDisk:
        return tlmv.SurfaceDisk(
            radius=self.display_diameter.item() / 2, matrix=matrix.tolist()
        )
