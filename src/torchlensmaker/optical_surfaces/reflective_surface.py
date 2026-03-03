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

from typing import Self
from torchlensmaker.types import BatchNDTensor
from torchlensmaker.optical_data import OpticalData
from torchlensmaker.core.ray_bundle import RayBundle
from torchlensmaker.elements.sequential import SequentialElement
from torchlensmaker.physics.physics_elements import ReflectiveInterface

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
        self.surface = surface
        self.propagator = SurfacePropagator(surface)
        self.reflector = SurfaceReflector()

    def reverse(self) -> Self:
        # TODO make a copy, surface should be a module
        return self

    def forward(self, data: OpticalData) -> OpticalData:
        rays_propagated, normals, fk_next = self.propagator(data.rays, data.fk)
        rays_reflected = self.reflector(rays_propagated, normals)

        return data.replace(rays=rays_reflected, fk=fk_next)
