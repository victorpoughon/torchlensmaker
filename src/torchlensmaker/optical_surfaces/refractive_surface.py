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

from typing import Self, Any
from torchlensmaker.types import TIRMode, BatchNDTensor, Direction
from torchlensmaker.core.ray_bundle import RayBundle
from torchlensmaker.optical_data import OpticalData
from torchlensmaker.surfaces.surface_element import SurfaceElement
from torchlensmaker.elements.sequential import SequentialElement
from torchlensmaker.core.base_module import BaseModule
from torchlensmaker.physics.physics_elements import RefractiveInterface
from torchlensmaker.materials.get_material_model import (
    MaterialModel,
    get_material_model,
)

from .surface_propagator import SurfacePropagator


class SurfaceRefractor(BaseModule):
    """
    Implements refraction at a surface boundary to reorient a ray bundle
    """

    def __init__(
        self,
        materials: tuple[str | MaterialModel, str | MaterialModel],
        tir_mode: TIRMode = "absorb",
    ):
        super().__init__()
        self.materials = (
            get_material_model(materials[0]).clone(),
            get_material_model(materials[1]).clone(),
        )
        self.tir_mode = tir_mode
        self.refractive_interface = RefractiveInterface()

    def forward(
        self, rays: RayBundle, normals: BatchNDTensor, direction: Direction
    ) -> RayBundle:
        # Compute indices of refraction
        n1 = self.materials[0](rays.wavel)
        n2 = self.materials[1](rays.wavel)

        if direction.is_retrograde():
            n1, n2 = n2, n1

        assert n1.shape == n2.shape == (rays.batch_size)
        assert n1.device == n2.device

        # Snell's law happens here
        refracted, valid_refraction = self.refractive_interface(rays.V, normals, n1, n2)

        if self.tir_mode == "reflect":
            new_rays = rays.reorient(refracted)
        else:
            new_rays = rays.reorient_absorb(refracted, valid_refraction)
            n2 = n2[valid_refraction]

        return new_rays


class RefractiveSurface(SequentialElement):
    def __init__(
        self,
        surface: SurfaceElement,
        materials: tuple[str | MaterialModel, str | MaterialModel],
        tir_mode: TIRMode = "absorb",
    ):
        super().__init__()
        self.propagator = SurfacePropagator(surface)
        self.refractor = SurfaceRefractor(materials, tir_mode)

    @property
    def surface(self) -> SurfaceElement:
        return self.propagator.surface

    @property
    def materials(self) -> MaterialModel:
        return self.refractor.materials

    @property
    def tir_mode(self) -> TIRMode:
        return self.refractor.tir_mode

    def clone(self, **overrides: Any) -> Self:
        kwargs = dict(
            surface=self.surface,
            materials=self.materials,
            tir_mode=self.refractor.tir_mode,
        )
        return type(self)(**kwargs | overrides)

    def forward(self, data: OpticalData) -> OpticalData:
        rays_propagated, normals, fk_next = self.propagator(
            data.rays, data.fk, data.direction
        )
        rays_refracted = self.refractor(rays_propagated, normals, data.direction)

        return data.replace(rays=rays_refracted, fk=fk_next)
