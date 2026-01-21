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
from jaxtyping import Float, Int

from torchlensmaker.core.tensor_manip import to_tensor

from .sampling_kernels import (
    LinspaceSampling1DKernel,
    LinspaceSampling2DKernel,
    ZeroSampling1DKernel,
    ZeroSampling2DKernel,
)


class ZeroSampler1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.kernel = ZeroSampling1DKernel()

    def forward(
        self, dtype: torch.dtype, device: torch.device
    ) -> Float[torch.Tensor, " N"]:
        return self.kernel.forward(dtype, device)


class ZeroSampler2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.kernel = ZeroSampling2DKernel()

    def forward(
        self, dtype: torch.dtype, device: torch.device
    ) -> Float[torch.Tensor, "N 2"]:
        return self.kernel.forward(dtype, device)


class LinspaceSampler1D(nn.Module):
    def __init__(self, N: Int[torch.Tensor, ""] | int):
        super().__init__()
        self.N = to_tensor(N, default_dtype=torch.int64)
        self.kernel = LinspaceSampling1DKernel()

    def forward(
        self, dtype: torch.dtype, device: torch.device
    ) -> Float[torch.Tensor, " N"]:
        return self.kernel.forward(self.N, dtype, device)


class LinspaceSampler2D(nn.Module):
    def __init__(
        self, Nx: Int[torch.Tensor, ""] | int, Ny: Int[torch.Tensor, ""] | int
    ):
        super().__init__()
        self.Nx = to_tensor(Nx, default_dtype=torch.int64)
        self.Ny = to_tensor(Ny, default_dtype=torch.int64)
        self.kernel = LinspaceSampling2DKernel()

    def forward(
        self, dtype: torch.dtype, device: torch.device
    ) -> Float[torch.Tensor, "N 2"]:
        return self.kernel.forward(self.Nx, self.Ny, dtype, device)
