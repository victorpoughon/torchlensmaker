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


class Sequential(nn.Sequential):
    def forward(self, data: OpticalData) -> OpticalData:
        for module in self:
            if hasattr(module, "sequential") and callable(module.sequential):
                data = module.sequential(data)
            else:
                data = module(data)
        return data
