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

from torchlensmaker.core.ray_bundle import RayBundle
from torchlensmaker.elements.sequential import SequentialElement
from torchlensmaker.optical_data import OpticalData
from torchlensmaker.physics.physics_elements import ReflectiveInterface
from torchlensmaker.surfaces.surface_element import SurfaceElement
from torchlensmaker.types import BatchNDTensor, Direction, Tf

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


class ReflectiveSurface(SequentialElement):
    def __init__(self, surface: nn.Module):
        super().__init__()
        self.propagator = SurfacePropagator(surface)
        self.reflector = SurfaceReflector()

    @property
    def surface(self) -> SurfaceElement:
        return self.propagator.surface

    def clone(self, **overrides: Any) -> Self:
        kwargs: dict[str, Any] = dict(surface=self.surface)
        return type(self)(**kwargs | overrides)

    def sequential(self, inputs: OpticalData) -> OpticalData:
        rays_reflected, tf_surface, fk_next = self(
            inputs.rays, inputs.fk, inputs.direction
        )
        return inputs.replace(rays=rays_reflected, fk=fk_next)

    def forward(
        self, rays: RayBundle, tf: Tf, direction: Direction
    ) -> tuple[RayBundle, Tf, Tf]:
        rays_propagated, normals, tf_surface, fk_next = self.propagator(
            rays, tf, direction
        )
        rays_reflected = self.reflector(rays_propagated, normals)

        return rays_reflected, tf_surface, fk_next
