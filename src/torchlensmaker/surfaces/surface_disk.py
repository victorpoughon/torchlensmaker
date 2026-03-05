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
import torch
import torch.nn as nn
from typing import Any, Self

from torchlensmaker.types import (
    ScalarTensor,
    BatchTensor,
    Batch2DTensor,
    Batch3DTensor,
    BatchNDTensor,
    MaskTensor,
    HomMatrix,
    Tf,
)
from .surface_element import SurfaceElement
from torchlensmaker.core.geometry import unit_vector
from torchlensmaker.kinematics.homogeneous_geometry import (
    hom_identity_2d,
    hom_identity_3d,
)

from torchlensmaker.core.tensor_manip import init_param
from torchlensmaker.core.functional_kernel import FunctionalKernel
from .raytrace import raytrace
from .sag_geometry import lens_diameter_domain_2d, lens_diameter_domain_3d
from .kernels_utils import example_rays_2d, example_rays_3d


def intersection_disk_2d(
    P: Batch2DTensor,
    V: Batch2DTensor,
    diameter: ScalarTensor,
) -> tuple[BatchTensor, Batch2DTensor, MaskTensor]:
    "2D ray intersection with the Y axis"

    t = -P[..., 0] / V[..., 0]
    normals = unit_vector(2, P.dtype, P.device).expand_as(V)

    points = P + t.unsqueeze(-1) * V
    valid = lens_diameter_domain_2d(points, diameter)
    return t, normals, valid


def intersection_disk_3d(
    P: Batch2DTensor,
    V: Batch2DTensor,
    diameter: ScalarTensor,
) -> tuple[BatchTensor, Batch3DTensor, MaskTensor]:
    "3D ray intersection with the YZ plane"

    t = -P[..., 0] / V[..., 0]
    normals = unit_vector(3, P.dtype, P.device).expand_as(V)

    points = P + t.unsqueeze(-1) * V
    valid = lens_diameter_domain_3d(points, diameter)
    return t, normals, valid


class DiskSurfaceKernel(FunctionalKernel):
    """
    Functional kernel for a disk perpendicular to the optical axis.
    In 2D, it reduces to a line segment, sometimes called a plane.
    """

    inputs = {
        "P": BatchNDTensor,
        "V": BatchNDTensor,
        "tf_in": Tf,
    }

    params = {
        "diameter": ScalarTensor,
    }

    outputs = {
        "t": BatchTensor,
        "normals": BatchNDTensor,
        "valid": MaskTensor,
        "surface_tf": Tf,
        "next_tf": Tf,
    }

    def __init__(self, dim: int):
        self.dim = dim

    def apply(
        self,
        P: BatchNDTensor,
        V: BatchNDTensor,
        tf_in: Tf,
        diameter: ScalarTensor,
    ) -> tuple[BatchTensor, BatchNDTensor, MaskTensor, Tf, Tf]:
        solver = intersection_disk_2d if self.dim == 2 else intersection_disk_3d
        local_solver = partial(solver, diameter=diameter)
        t, normals, valid = raytrace(P, V, tf_in, local_solver)
        return t, normals, valid, tf_in.clone(), tf_in.clone()

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

    def example_params(
        self, dtype: torch.dtype, device: torch.device
    ) -> tuple[ScalarTensor]:
        return (torch.tensor(10.0, dtype=dtype, device=device),)


class Disk(SurfaceElement):
    """
    Disk surface (2D or 3D)
    """

    def __init__(self, diameter: float | ScalarTensor):
        super().__init__()
        self.diameter = init_param(self, "diameter", diameter, False)
        self.func2d = DiskSurfaceKernel(2)
        self.func3d = DiskSurfaceKernel(3)

    def clone(self, **overrides: Any) -> Self:
        kwargs = dict(diameter=self.diameter)
        return type(self)(**kwargs | overrides)

    def forward(
        self, P: BatchNDTensor, V: BatchNDTensor, tf: Tf
    ) -> tuple[BatchTensor, BatchNDTensor, MaskTensor, Tf, Tf]:
        func = self.func2d if P.shape[-1] == 2 else self.func3d
        return func.apply(P, V, tf, self.diameter)

    def outer_extent(self, r: ScalarTensor) -> ScalarTensor | None:
        return torch.zeros_like(r)

    def render(self) -> Any:
        return {
            "type": "surface-plane",
            "radius": self.diameter.item() / 2,
        }
