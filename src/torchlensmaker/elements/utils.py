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


class Marker(nn.Module):
    "WIP"

    def __init__(self, text: str):
        super().__init__()
        self.text = text

    def forward(self, inputs: OpticalData) -> OpticalData:
        return inputs


class MixedDim(nn.Module):
    "2D or 3D branch"

    def __init__(self, dim2: nn.Module, dim3: nn.Module):
        super().__init__()
        self._dim2 = dim2
        self._dim3 = dim3

    def sequential(self, data: OpticalData) -> OpticalData:
        if data.dim == 2:
            return self._dim2.sequential(data)
        else:
            return self._dim3.sequential(data)
