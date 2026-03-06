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


import torch
from typing import Any, Self
from jaxtyping import Float
from torchlensmaker.types import (
    ScalarTensor,
    BatchTensor,
    Tf,
    BatchNDTensor,
    MaskTensor,
)
from torchlensmaker.core.tensor_manip import init_param
from torchlensmaker.core.base_module import BaseModule
from torchlensmaker.surfaces.surface_element import SurfaceElement
from torchlensmaker.surfaces.sag_geometry import (
    anchor_transforms_2d,
    anchor_transforms_3d,
)


class SurfaceAnchor(BaseModule):
    """
    apply tf: entrance anchor + scale
    eval surface
    unapply tf: entrace anchor + scale
    apply exit anchor

    Surface must support
        outer_extent
        diameter
    """

    def __init__(
        self,
        surface: SurfaceElement,
        anchors: tuple[float, float] | Float[torch.Tensor, " 2"] = (0.0, 0.0),
        scale: float | ScalarTensor = 1.0,
    ):
        super().__init__()
        self.surface = surface.clone()
        self.anchors = init_param(self, "anchors", anchors, False)
        self.scale = init_param(self, "scale", scale, False)

    def clone(self, **overrides: Any) -> Self:
        kwargs = dict(surface=self.surface, anchors=self.anchors, scale=self.scale)
        return type(self)(**kwargs | overrides)

    def render(self) -> Any:
        return self.surface.render()

    def forward(
        self, P: BatchTensor, V: BatchTensor, tf: Tf
    ) -> tuple[BatchTensor, BatchNDTensor, MaskTensor, Tf, Tf]:
        dim = P.shape[-1]

        extent0 = self.surface.outer_extent(self.anchors[0] * self.surface.diameter / 2)
        extent1 = self.surface.outer_extent(self.anchors[1] * self.surface.diameter / 2)

        anchor_function = anchor_transforms_2d if dim == 2 else anchor_transforms_3d
        tf_surface, tf_next = anchor_function(extent0, extent1, self.scale, tf)

        t, normals, valid, _, _ = self.surface(P, V, tf_surface)

        return t, normals, valid, tf_surface, tf_next
