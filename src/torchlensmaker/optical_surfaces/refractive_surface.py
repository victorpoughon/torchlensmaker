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
from torchlensmaker.types import TIRMode
from torchlensmaker.optical_data import OpticalData
from torchlensmaker.core.ray_bundle import RayBundle
from torchlensmaker.elements.sequential import SequentialElement
from torchlensmaker.physics.physics_elements import RefractiveInterface
from torchlensmaker.materials.get_material_model import (
    MaterialModel,
    get_material_model,
)


class RefractiveSurface(SequentialElement):
    def __init__(
        self,
        surface: nn.Module,
        material: str | MaterialModel,
        tir_mode: TIRMode = "absorb",
    ):
        super().__init__()
        self.surface = surface
        self.refractive_interface = RefractiveInterface()
        self.material = get_material_model(material)
        self._tir_mode = tir_mode

    def reverse(self) -> Self:
        # TODO make a copy, surface should be a module
        return self

    def forward(self, data: OpticalData) -> OpticalData:
        # Raytrace with the surface
        t, normals, valid_collision, fk_surface, fk_next = self.surface(
            data.rays.P, data.rays.V, data.fk
        )

        # First step: propagate rays forward to their collision point
        # This produces a new ray bundle, with possibly fewer rays
        rays_step1 = data.rays.propagate_absorb(t, valid_collision)
        normals = normals[valid_collision]

        ########

        # Compute indices of refraction
        n1 = rays_step1.index
        n2 = self.material(rays_step1.wavel)
        assert n1.shape == n2.shape == (rays_step1.batch_size)

        # Snell's law happens here
        # Compute refraction on the full frame rays (including non-colliding
        # rays), so that comparing the two valid masks is easier
        refracted, valid_refraction = self.refractive_interface(
            rays_step1.V, normals, n1, n2
        )

        if self._tir_mode == "reflect":
            rays_step2 = rays_step1.reorient(refracted)
        else:
            rays_step2 = rays_step1.reorient_absorb(refracted, valid_refraction)
            n2 = n2[valid_refraction]

        # Apply the new index of refraction to the new bundle
        rays_step3 = rays_step2.replace(index=n2)

        return data.replace(rays=rays_step3, fk=fk_next)
