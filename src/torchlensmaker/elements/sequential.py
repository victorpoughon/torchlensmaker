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

from typing import Type, Sequence, Self
from collections import OrderedDict
import torch.nn as nn
from torchlensmaker.optical_data import OpticalData

from torchlensmaker.elements.utils import (
    get_elements_by_type,
)

from torchlensmaker.light_sources.light_sources_query import (
    set_sampling2d,
    set_sampling3d,
)

from .sequential_element import SequentialElement


class SubChain(SequentialElement):
    def __init__(self, *children: nn.Module):
        super().__init__()
        self._sequential = Sequential(*children)

    def forward(self, inputs: OpticalData) -> OpticalData:
        output: OpticalData = self._sequential(inputs)
        return output.replace(dfk=inputs.dfk, ifk=inputs.ifk)


class Sequential(nn.Sequential, SequentialElement):
    def forward(self, data: OpticalData) -> OpticalData:
        for module in self:
            data = module.sequential(data)
        return data

    def get_elements_by_type(self, typ: Type[nn.Module]) -> nn.ModuleList:
        return get_elements_by_type(self, typ)

    def reverse(self) -> Self:
        return type(self)(
            OrderedDict(
                reversed(
                    list((name, mod.reverse()) for (name, mod) in self.named_children())
                )
            )
        )

    def set_sampling2d(
        self,
        pupil: int | Sequence[float] | None = None,
        field: int | Sequence[float] | None = None,
        wavelength: int | Sequence[float] | None = None,
    ) -> None:
        return set_sampling2d(self, pupil, field, wavelength)

    def set_sampling3d(
        self,
        pupil: int | Sequence[float] | None = None,
        field: int | Sequence[float] | None = None,
        wavelength: int | Sequence[float] | None = None,
    ) -> None:
        return set_sampling3d(self, pupil, field, wavelength)
