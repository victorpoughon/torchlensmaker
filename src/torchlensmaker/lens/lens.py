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

from jaxtyping import Float
import torch
import torch.nn as nn

from torchlensmaker.elements.sequential import SequentialElement, Sequential
from torchlensmaker.optical_data import OpticalData
from .lens_thickness import (
    lens_inner_thickness,
    lens_outer_thickness,
    lens_minimal_diameter,
)


class Lens(SequentialElement):
    def __init__(self, *sequence: nn.Module):
        super().__init__()
        self.sequence = Sequential(*sequence)

    def forward(self, data: OpticalData) -> OpticalData:
        return self.sequence(data)

    def inner_thickness(self) -> Float[torch.Tensor, ""]:
        return lens_inner_thickness(self)

    def outer_thickness(self) -> Float[torch.Tensor, ""]:
        return lens_outer_thickness(self)

    def minimal_diameter(self) -> Float[torch.Tensor, ""]:
        return lens_minimal_diameter(self)
