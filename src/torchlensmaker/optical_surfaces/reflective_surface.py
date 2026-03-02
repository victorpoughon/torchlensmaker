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

from typing import Sequence, Optional, TypeAlias, Literal, Self
from torchlensmaker.types import BatchTensor, MaskTensor, BatchNDTensor, MissMode
from torchlensmaker.optical_data import OpticalData, propagate
from torchlensmaker.elements.sequential import SequentialElement
from torchlensmaker.physics.physics_elements import ReflectiveInterface


class ReflectiveSurface(SequentialElement):
    def __init__(self, surface: nn.Module, miss_mode: MissMode = "absorb"):
        super().__init__()
        self.surface = surface
        self.reflective_interface = ReflectiveInterface()
        self._miss_mode = miss_mode

    def reverse(self) -> Self:
        # TODO make a copy, surface should be a module
        return self

    def forward(self, data: OpticalData) -> OpticalData:
        t, normals, valid, fk_surface, fk_next = self.surface(data.rays.P, data.rays.V, data.fk)
        reflected = self.reflective_interface(data.rays.V[valid], normals[valid])

        propagated = propagate(data, t, valid, reflected, self._miss_mode)

        return propagated.replace(fk=fk_next)
