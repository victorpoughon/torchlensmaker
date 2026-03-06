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
import torch.nn as nn

from torchlensmaker.types import Tf, BatchNDTensor
from torchlensmaker.core.base_module import BaseModule
from torchlensmaker.core.ray_bundle import RayBundle


class SurfacePropagator(BaseModule):
    """
    Wraps a SurfaceElement into an element that propagates a ray bundle
    """

    def __init__(self, surface: nn.Module):
        super().__init__()
        self.surface = surface.clone()

    def forward(
        self, rays: RayBundle, tf: Tf, reverse: bool
    ) -> tuple[RayBundle, BatchNDTensor, Tf]:
        # Raytrace with the surface
        t, normals, valid_collision, fk_surface, fk_next = self.surface(
            rays.P, rays.V, tf, reverse
        )

        # Propagate rays forward to their collision point
        # This produces a new ray bundle, with possibly fewer rays
        new_rays = rays.propagate_absorb(t, valid_collision)
        normals = normals[valid_collision]
        return new_rays, normals, fk_next
