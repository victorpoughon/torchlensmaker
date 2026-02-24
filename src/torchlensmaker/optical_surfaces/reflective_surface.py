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

from torchlensmaker.optical_data import OpticalData
from torchlensmaker.elements.sequential import SequentialElement
from torchlensmaker.physics.physics_elements import ReflectiveInterface


MissMode: TypeAlias = Literal["absorb", "pass", "error"]


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
        t, normals, valid, fk_surface, fk_next = self.surface(data.P, data.V, data.fk)
        reflected = self.reflective_interface(data.V[valid], normals[valid])

        # "ray advancer": given everything it needs, construct the next ray
        # bundle, changing origin points and respecting masks

        collision_points = data.P + t.unsqueeze(1).expand_as(data.V) * data.V

        if self._miss_mode == "absorb":
            # return hits only
            return data.filter_variables(valid).replace(
                P=collision_points[valid], V=reflected, fk=fk_next
            )

        elif self._miss_mode == "pass":
            # insert hit rays as reflected
            return data.replace(
                P=data.P.masked_scatter(valid.unsqueeze(-1), collision_points[valid]),
                V=data.V.masked_scatter(valid.unsqueeze(-1), reflected),
                fk=fk_next,
            )

        elif self._miss_mode == "error":
            misses = (~valid).sum()
            if misses != 0:
                raise RuntimeError(
                    f"Some rays ({misses}) don't collide with surface, but miss option is '{self._miss_mode}'"
                )
            # return all rays as hits
            return data.replace(P=collision_points, V=reflected, fk=fk_next)
