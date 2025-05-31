# This file is part of Torch Lens Maker
# Copyright (C) 2025 Victor Poughon
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

import torch.nn as nn

from torchlensmaker.optical_data import OpticalData
from torchlensmaker.elements.kinematics import KinematicElement

from typing import overload
from collections import OrderedDict


class SequentialElement(nn.Module):
    """Base class for sequential elements"""

    def forward(self, data: OpticalData) -> OpticalData:
        raise NotImplementedError


class SequentialKinematicElement(SequentialElement):
    "Wraps a KinematicElement into a SequentialElement"

    def __init__(self, element: KinematicElement):
        super().__init__()
        self._element = element

    def forward(self, data: OpticalData) -> OpticalData:
        return data.replace(transforms=self(data.transforms))


class SubChain(SequentialElement):
    def __init__(self, *children: nn.Module):
        super().__init__()
        self._sequential = Sequential(*children)

    def forward(self, inputs: OpticalData) -> OpticalData:
        output: OpticalData = self._sequential(inputs)
        new_chain = inputs.transforms
        return output.replace(transforms=new_chain)


def sequential_wrap(element: nn.Module | None) -> SequentialElement:
    "Wrap element to make it sequential"
    if isinstance(SequentialElement, element):
        return element
    
    if isinstance(element, KinematicElement):
        return SequentialKinematicElement(element)
    else:
        raise RuntimeError(f"Cannot use {element} in a Sequential model")


class Sequential(SequentialElement):
    @overload
    def __init__(self, *args: nn.Module) -> None:
        ...

    @overload
    def __init__(self, arg: 'OrderedDict[str, nn.Module]') -> None:
        ...

    def __init__(self, *args):
        super().__init__()
    
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                wrapped = sequential_wrap(module)
                self.add_module(key, wrapped)
        else:
            for idx, module in enumerate(args):
                wrapped = sequential_wrap(module)
                self.add_module(str(idx), wrapped)
