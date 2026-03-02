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
from torchlensmaker.types import MissMode, TIRMode
from torchlensmaker.optical_data import OpticalData, propagate
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
        miss_mode: MissMode = "absorb",
        tir_mode: TIRMode = "absorb",
    ):
        super().__init__()
        self.surface = surface
        self.refractive_interface = RefractiveInterface()
        self.material = get_material_model(material)
        self._miss_mode = miss_mode
        self._tir_mode = tir_mode

    def reverse(self) -> Self:
        # TODO make a copy, surface should be a module
        return self

    def forward(self, data: OpticalData) -> OpticalData:
        # Compute indices of refraction
        n1 = data.rays.index
        n2 = self.material(data.rays.wavel)
        assert n1.shape == n2.shape == (data.rays.P.shape[0],)

        # Raytrace with the surface
        t, normals, valid_collision, fk_surface, fk_next = self.surface(
            data.rays.P, data.rays.V, data.fk
        )

        # Snell's law happens here
        # Compute refraction on the full frame rays (including non-colliding
        # rays), so that comparing the two valid masks is easier
        refracted, valid_refraction = self.refractive_interface(
            data.rays.V, normals, n1, n2
        )

        if self._tir_mode == "reflect":
            valid = valid_collision
        else:
            valid = torch.logical_and(valid_collision, valid_refraction)

        propagated = propagate(data, t, valid, refracted[valid], self._miss_mode)
        return propagated.replace(
            rays=propagated.rays.replace(index=n2[valid]), fk=fk_next
        )
