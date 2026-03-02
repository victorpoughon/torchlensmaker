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
from torchlensmaker.types import ScalarTensor
from torchlensmaker.optical_data import OpticalData
from torchlensmaker.elements.sequential import SequentialElement
from torchlensmaker.physics.physics_elements import ReflectiveInterface
from torchlensmaker.surfaces.surface_disk import Disk


class Aperture(SequentialElement):
    def __init__(self, diameter: float | ScalarTensor):
        super().__init__()
        self.surface = Disk(diameter)

    def forward(self, data: OpticalData) -> OpticalData:
        # Collision detection with the surface
        t, _, valid_collision, tf_surface, tf_next = self.surface(
            data.rays.P, data.rays.V, data.fk
        )

        # Keep colliding rays only
        new_rays = data.rays.propagate_absorb(t, valid_collision)

        return data.replace(
            rays=new_rays,
            fk=tf_next,  # correct but useless cause Aperture is only circular plane currently
        )

    def reverse(self) -> Self:
        # TODO make a copy, surface should be a module
        return self
