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

from enum import Enum
import torch.nn as nn
from torchlensmaker.optical_data import OpticalData


class Dim(Enum):
    """
    Enum to represent the physical dimensionality of a model
    can be:
        * TWO: the model represents two physical dimensions
        * THREE: the model represents three physical dimensions
        * MIXED: the model represents two or three physical dimensions
    """

    ONE = 1
    TWO = 2
    THREE = 3
    MIXED = 4


class SequentialElement(nn.Module):
    """
    Base class for sequential elements

    A sequential element is an element that can be used in a Sequential model,
    because it provides a sequential() forward method.
    """

    def sequential(self, data: OpticalData) -> OpticalData:
        # default implementation just calls forward, can be overwritten
        return self(data)

    def reverse(self) -> "SequentialElement":
        raise NotImplementedError(
            f"reverse() method not implemented for type {type(self).__name__}"
        )


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
