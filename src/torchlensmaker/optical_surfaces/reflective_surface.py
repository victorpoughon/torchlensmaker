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
import torch.nn as nn

from torchlensmaker.core.base_module import BaseModule
from torchlensmaker.core.ray_bundle import RayBundle
from torchlensmaker.elements.sequential_data import SequentialData
from torchlensmaker.physics.physics_elements import ReflectiveInterface
from torchlensmaker.surfaces.surface_element import SurfaceElement
from torchlensmaker.types import BatchNDTensor, Tf

from .optical_surface import OpticalSurfaceElement
from .surface_propagator import SurfacePropagator


class SurfaceReflector(nn.Module):
    """
    Implements reflection at a surface boundary to reorient a ray bundle
    """

    def __init__(self):
        super().__init__()
        self.reflective_interface = ReflectiveInterface()

    def forward(self, rays: RayBundle, normals: BatchNDTensor) -> RayBundle:
        reflected = self.reflective_interface(rays.V, normals)
        return rays.reorient(reflected)


class ReflectiveSurface(OpticalSurfaceElement):
    def __init__(self, surface: SurfaceElement):
        super().__init__()
        self.propagator = SurfacePropagator(surface)
        self.reflector = SurfaceReflector()

    @property
    def surface(self) -> SurfaceElement:
        return self.propagator.surface

    def clone(self, **overrides: Any) -> Self:
        kwargs: dict[str, Any] = dict(surface=self.surface)
        return type(self)(**kwargs | overrides)

    def reverse(self) -> Self:
        return self.clone(surface=self.surface.reverse())

    def forward(self, rays: RayBundle, tf: Tf) -> tuple[RayBundle, Tf, Tf]:
        rays_propagated, normals, tf_surface, fk_next = self.propagator(rays, tf)
        rays_reflected = self.reflector(rays_propagated, normals)

        return rays_reflected, tf_surface, fk_next
