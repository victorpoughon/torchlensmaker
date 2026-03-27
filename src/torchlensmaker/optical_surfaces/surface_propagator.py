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

from torchlensmaker.core.ray_bundle import RayBundle
from torchlensmaker.surfaces.surface_element import SurfaceElement
from torchlensmaker.types import BatchNDTensor, BatchTensor, MaskTensor, Tf


class SurfacePropagator(nn.Module):
    """
    Wraps a SurfaceElement into an element that propagates a ray bundle
    """

    def __init__(self, surface: SurfaceElement):
        super().__init__()
        self.surface = surface.clone()

    def forward(
        self, rays: RayBundle, tf: Tf
    ) -> tuple[RayBundle, BatchTensor, BatchNDTensor, MaskTensor, Tf, Tf]:
        # Raytrace with the surface
        surface_outputs = self.surface(
            rays.P,
            rays.V,
            tf,
        )

        # First step: propagate rays forward to their collision point
        # This produces a new ray bundle, with possibly fewer rays
        new_rays = rays.propagate_absorb(surface_outputs.t, surface_outputs.valid)
        normals = surface_outputs.normals[surface_outputs.valid]
        return (
            new_rays,
            surface_outputs.t,
            normals,
            surface_outputs.valid,
            surface_outputs.tf_surface,
            surface_outputs.tf_next,
        )
