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
from jaxtyping import Float

from torchlensmaker.core.base_module import BaseModule
from torchlensmaker.core.functional_kernel import FunctionalKernel
from torchlensmaker.core.tensor_manip import init_param
from torchlensmaker.kinematics.homogeneous_geometry import (
    hom_identity_2d,
    hom_identity_3d,
)
from torchlensmaker.surfaces.sag_geometry import (
    anchor_transforms_2d,
    anchor_transforms_3d,
)
from torchlensmaker.surfaces.surface_element import SurfaceElement
from torchlensmaker.types import (
    BatchNDTensor,
    BatchTensor,
    MaskTensor,
    ScalarTensor,
    Tf,
)


class SurfaceScaleAnchorKernel(FunctionalKernel):
    """
    Compute transforms that enable scaling and anchoring a surface

    Inputs:
        extent0: X value of the first anchor point
        extent1: X value of the second anchor point
        scale: scale factor that applies to the surface (1 or -1)

    Outputs:
        tf_surface: transform that applies to the surface
        tf_next: transform that applies to the next joint in the kinematic chain
    """

    inputs = {
        "extent0": ScalarTensor,
        "extent1": ScalarTensor,
        "scale": ScalarTensor,
        "tf": Tf,
    }
    params = {}
    outputs = {"tf_surface": Tf, "tf_next": Tf}

    def __init__(self, dim: int):
        self.dim = dim

    def apply(
        self, extent0: ScalarTensor, extent1: ScalarTensor, scale: ScalarTensor, tf: Tf
    ) -> tuple[Tf, Tf]:
        anchor_function = (
            anchor_transforms_2d if self.dim == 2 else anchor_transforms_3d
        )
        return anchor_function(extent0, extent1, scale, tf)

    def example_inputs(
        self, dtype: torch.dtype, device: torch.device
    ) -> tuple[ScalarTensor, ScalarTensor, ScalarTensor, Tf]:
        tf_id = hom_identity_2d if self.dim == 2 else hom_identity_3d
        return (
            torch.tensor(0.0, dtype=dtype, device=device),
            torch.tensor(1.0, dtype=dtype, device=device),
            torch.tensor(1.0, dtype=dtype, device=device),
            tf_id(dtype, device),
        )

    def example_params(self, dtype: torch.dtype, device: torch.device) -> tuple[()]:
        return tuple()


class KinematicSurface(BaseModule):
    """
    Apply anchor and scale to a wrapped surface element
    """

    def __init__(
        self,
        surface: SurfaceElement,
        anchors: tuple[float, float] | Float[torch.Tensor, " 2"] = (0.0, 0.0),
        scale: float | ScalarTensor = 1.0,
    ):
        super().__init__()
        self.surface = surface
        self.anchors = init_param(self, "anchors", anchors, False)
        self.scale = init_param(self, "scale", scale, False)
        self.kernel_anchor2d = SurfaceScaleAnchorKernel(2)
        self.kernel_anchor3d = SurfaceScaleAnchorKernel(3)

    def clone(self, **overrides: Any) -> Self:
        kwargs: dict[str, Any] = dict(
            surface=self.surface, anchors=self.anchors, scale=self.scale
        )
        return type(self)(**kwargs | overrides)

    def reverse(self) -> Self:
        return self.clone(anchors=self.anchors.flip(0))

    def render(self, matrix: torch.Tensor) -> Any:
        return self.surface.render(matrix)

    def forward(
        self, P: BatchTensor, V: BatchTensor, tf: Tf
    ) -> tuple[BatchTensor, BatchNDTensor, MaskTensor, Tf, Tf]:
        dim = P.shape[-1]

        kernel_anchor = self.kernel_anchor2d if tf.pdim() == 2 else self.kernel_anchor3d

        extent0 = self.surface.outer_extent(self.anchors[0])
        extent1 = self.surface.outer_extent(self.anchors[1])

        tf_surface, tf_next = kernel_anchor.apply(extent0, extent1, self.scale, tf)
        t, normals, valid, _, _ = self.surface(P, V, tf_surface)

        return t, normals, valid, tf_surface, tf_next
