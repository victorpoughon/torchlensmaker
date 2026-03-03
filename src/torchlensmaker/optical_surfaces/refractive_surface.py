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
from torchlensmaker.types import TIRMode, Tf, BatchNDTensor
from torchlensmaker.optical_data import OpticalData
from torchlensmaker.core.ray_bundle import RayBundle
from torchlensmaker.elements.sequential import SequentialElement
from torchlensmaker.physics.physics_elements import RefractiveInterface
from torchlensmaker.materials.get_material_model import (
    MaterialModel,
    get_material_model,
)

from .surface_propagator import SurfacePropagator


class SurfaceRefractor(nn.Module):
    """
    Implements refraction at a surface boundary to reorient a ray bundle
    """

    def __init__(
        self,
        material: str | MaterialModel,
        tir_mode: TIRMode = "absorb",
    ):
        super().__init__()
        self.material = get_material_model(material)
        self._tir_mode = tir_mode
        self.refractive_interface = RefractiveInterface()

    def forward(self, rays: RayBundle, normals: BatchNDTensor) -> RayBundle:
        # Compute indices of refraction
        n1 = rays.index
        n2 = self.material(rays.wavel)
        assert n1.shape == n2.shape == (rays.batch_size)

        # Snell's law happens here
        refracted, valid_refraction = self.refractive_interface(rays.V, normals, n1, n2)

        if self._tir_mode == "reflect":
            new_rays = rays.reorient(refracted)
        else:
            new_rays = rays.reorient_absorb(refracted, valid_refraction)
            n2 = n2[valid_refraction]

        # Also apply the new index of refraction to the new bundle
        return new_rays.replace(index=n2)


class RefractiveSurface(SequentialElement):
    def __init__(
        self,
        surface: nn.Module,
        material: str | MaterialModel,
        tir_mode: TIRMode = "absorb",
    ):
        super().__init__()
        self.surface = surface
        self.propagator = SurfacePropagator(surface)
        self.refractor = SurfaceRefractor(material, tir_mode)

    def reverse(self) -> Self:
        # TODO make a copy, surface should be a module
        return self

    def forward(self, data: OpticalData) -> OpticalData:
        rays_propagated, normals, fk_next = self.propagator(data.rays, data.fk)
        rays_refracted = self.refractor(rays_propagated, normals)

        return data.replace(rays=rays_refracted, fk=fk_next)
