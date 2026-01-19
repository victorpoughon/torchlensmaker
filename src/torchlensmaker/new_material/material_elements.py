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
from jaxtyping import Float

from torchlensmaker.core.tensor_manip import to_tensor

from .material_kernels import NonDispersiveMaterialKernel


class NonDispersiveMaterial(nn.Module):
    def __init__(self, n: Float[torch.Tensor, ""]):
        super().__init__()
        self.n = to_tensor(n)
        self.kernel = NonDispersiveMaterialKernel()

    def forward(
        self, wavelength: Float[torch.Tensor, " N"]
    ) -> Float[torch.Tensor, " N"]:
        return self.kernel.forward(wavelength, self.n)
