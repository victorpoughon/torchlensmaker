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
from typing import Any

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
    hom_identity_2d,
)

from torchlensmaker.core.tensor_manip import init_param
from torchlensmaker.core.functional_kernel import FunctionalKernel
from .raytrace import raytrace
from .sag_geometry import lens_diameter_domain_2d
from .kernels_utils import example_rays_2d


def intersection_disk_2d(
    diameter: ScalarTensor,
    P: Batch2DTensor,
    V: Batch2DTensor,
) -> tuple[BatchTensor, Batch2DTensor]:
    "Ray intersection with the Y axis"

    t = -P[..., 0] / V[..., 0]
    normals = torch.tensor(((1.0, 0.0)), dtype=V.dtype, device=V.device).expand_as(V)

    points = P + t.unsqueeze(-1) * V
    valid = lens_diameter_domain_2d(points, diameter)
    return t, normals, valid


class Disk2DSurfaceKernel(FunctionalKernel):
    """
    Functional kernel for a 2D disk (aka a line segment perpendicular to the
    optical axis), parameterized by lens diameter.
    """

    inputs = {
        "P": Batch2DTensor,
        "V": Batch2DTensor,
        "tf_in": Tf2D,
    }

    params = {
        "diameter": ScalarTensor,
    }

    outputs = {
        "t": BatchTensor,
        "normals": Batch2DTensor,
        "valid": MaskTensor,
        "surface_tf": Tf2D,
        "next_tf": Tf2D,
    }

    def apply(
        self,
        P: Batch2DTensor,
        V: Batch2DTensor,
        tf_in: Tf2D,
        diameter: ScalarTensor,
    ) -> tuple[BatchTensor, Batch2DTensor, MaskTensor, Tf2D, Tf2D]:
        # Perform raytrace
        local_solver = partial(intersection_disk_2d, diameter=diameter)
        t, normals, valid = raytrace(P, V, tf_in, local_solver)

        return t, normals, valid, tf_in.clone(), tf_in.clone()

    def example_inputs(
        self, dtype: torch.dtype, device: torch.device
    ) -> tuple[Batch2DTensor, Batch2DTensor, Tf2D]:
        P, V = example_rays_2d(10, dtype, device)
        tf = hom_identity_2d(dtype, device)
        return P, V, tf

    def example_params(
        self, dtype: torch.dtype, device: torch.device
    ) -> tuple[ScalarTensor]:
        return (torch.tensor(10.0, dtype=dtype, device=device),)


class Disk(nn.Module):
    """
    Disk surface (2D or 3D)
    """

    def __init__(self, diameter: float | ScalarTensor):
        super().__init__()
        self.diameter = init_param(self, "diameter", diameter, False)
        self.func2d = Disk2DSurfaceKernel()

    def forward(
        self, P: BatchTensor, V: BatchTensor, tf: Tf2D
    ) -> tuple[BatchTensor, BatchNDTensor, MaskTensor, Tf2D, Tf2D]:
        return self.func2d.apply(P, V, tf, self.diameter)

    def render(self) -> Any:
        return {
            "type": "surface-plane",
            "radius": self.diameter.item() / 2,
        }
