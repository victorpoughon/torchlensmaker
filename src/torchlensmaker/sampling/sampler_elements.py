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
    ExactSampling1DKernel,
    ExactSampling2DKernel,
    ZeroSampling1DKernel,
    ZeroSampling2DKernel,
    DiskSampling2DKernel,
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


class ExactSampler1D(nn.Module):
    def __init__(self, samples: Float[torch.Tensor, " N"]):
        super().__init__()
        self.samples = samples
        self.kernel = ExactSampling1DKernel()

    def __repr__(self) -> str:
        return f"{self._get_name()}(samples=<tensor of shape {self.samples.shape}>)"

    def forward(
        self, dtype: torch.dtype, device: torch.device
    ) -> Float[torch.Tensor, " N"]:
        return self.kernel.forward(self.samples, dtype, device)


class ExactSampler2D(nn.Module):
    def __init__(self, samples: Float[torch.Tensor, "N 2"]):
        super().__init__()
        self.samples = samples
        self.kernel = ExactSampling2DKernel()

    def __repr__(self) -> str:
        return f"{self._get_name()}(samples=<tensor of shape {self.samples.shape}>)"

    def forward(
        self, dtype: torch.dtype, device: torch.device
    ) -> Float[torch.Tensor, "N 2"]:
        return self.kernel.forward(self.samples, dtype, device)


class LinspaceSampler1D(nn.Module):
    def __init__(self, N: Int[torch.Tensor, ""] | int):
        super().__init__()
        self.N = to_tensor(N, default_dtype=torch.int64)
        self.kernel = LinspaceSampling1DKernel()

    def __repr__(self) -> str:
        return f"{self._get_name()}(N={self.N})"

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

    def __repr__(self) -> str:
        return f"{self._get_name()}(Nx={self.Nx}, Ny={self.Ny})"

    def forward(
        self, dtype: torch.dtype, device: torch.device
    ) -> Float[torch.Tensor, "N 2"]:
        return self.kernel.forward(self.Nx, self.Ny, dtype, device)


class DiskSampler2D(nn.Module):
    def __init__(
        self, Nrho: Int[torch.Tensor, ""] | int, Ntheta: Int[torch.Tensor, ""] | int
    ):
        super().__init__()
        self.Nrho = to_tensor(Nrho, default_dtype=torch.int64)
        self.Ntheta = to_tensor(Ntheta, default_dtype=torch.int64)
        self.kernel = DiskSampling2DKernel

    def __repr__(self) -> str:
        return f"{self._get_name()}(Nrho={self.Nrho}, Ntheta={self.Ntheta})"

    def forward(
        self, dtype: torch.dtype, device: torch.device
    ) -> Float[torch.Tensor, "N 2"]:
        return self.kernel.forward(self.Nrho, self.Ntheta, dtype, device)
