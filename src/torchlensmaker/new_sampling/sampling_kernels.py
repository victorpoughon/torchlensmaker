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

from typing import TypeAlias
from jaxtyping import Float, Int
import torch
from torchlensmaker.core.functional_kernel import FunctionalKernel

from .sampling import disk_sampling


class ZeroSampling1DKernel(FunctionalKernel):
    input_names = []
    param_names = []
    output_names = ["samples"]
    forward_dtype_device = True

    @staticmethod
    def forward(dtype: torch.dtype, device: torch.device) -> Float[torch.Tensor, " 1"]:
        return torch.zeros((1), dtype=dtype, device=device)

    @staticmethod
    def example_inputs(
        dtype: torch.dtype, device: torch.device
    ) -> tuple[torch.Tensor, ...]:
        return tuple()

    @staticmethod
    def example_params(
        dtype: torch.dtype, device: torch.device
    ) -> tuple[torch.Tensor, ...]:
        return tuple()


class ZeroSampling2DKernel(FunctionalKernel):
    input_names = []
    param_names = []
    output_names = ["samples"]
    forward_dtype_device = True

    @staticmethod
    def forward(dtype: torch.dtype, device: torch.device) -> Float[torch.Tensor, "1 2"]:
        return torch.zeros((1, 2), dtype=dtype, device=device)

    @staticmethod
    def example_inputs(
        dtype: torch.dtype, device: torch.device
    ) -> tuple[torch.Tensor, ...]:
        return tuple()

    @staticmethod
    def example_params(
        dtype: torch.dtype, device: torch.device
    ) -> tuple[torch.Tensor, ...]:
        return tuple()


class ExactSampling1DKernel(FunctionalKernel):
    input_names = []
    param_names = ["samples"]
    output_names = ["samples"]
    forward_dtype_device = True

    @staticmethod
    def forward(
        samples: Float[torch.Tensor, " N"], dtype: torch.dtype, device: torch.device
    ) -> Float[torch.Tensor, "1 2"]:
        return samples.to(dtype=dtype, device=device)

    @staticmethod
    def example_inputs(
        dtype: torch.dtype, device: torch.device
    ) -> tuple[torch.Tensor, ...]:
        return tuple()

    @staticmethod
    def example_params(
        dtype: torch.dtype, device: torch.device
    ) -> tuple[torch.Tensor, ...]:
        return tuple(torch.tensor([-0.5, 0.0, 0.5]))


class LinspaceSampling1DKernel(FunctionalKernel):
    input_names = []
    param_names = ["N"]
    output_names = ["samples"]
    forward_dtype_device = True
    export_legacy = True

    @staticmethod
    def forward(
        N: Int[torch.Tensor, ""], dtype: torch.dtype, device: torch.device
    ) -> Float[torch.Tensor, " N"]:
        one = torch.ones((), dtype=dtype, device=device)
        # note: extra .to(dtype=) seems required for onnx export dtype correctness
        # seems like there is some dependence on torch default dtype inside linspace even when dtype argument is provided

        samples = torch.linspace(-one, one, N, dtype=dtype, device=device).to(
            dtype=dtype
        )
        return samples

    @staticmethod
    def example_inputs(
        dtype: torch.dtype, device: torch.device
    ) -> tuple[torch.Tensor, ...]:
        return tuple()

    @staticmethod
    def example_params(
        dtype: torch.dtype, device: torch.device
    ) -> tuple[torch.Tensor, ...]:
        return (torch.tensor(10, dtype=torch.int64),)


class LinspaceSampling2DKernel(FunctionalKernel):
    input_names = []
    param_names = ["Nx", "Ny"]
    output_names = ["samples"]
    forward_dtype_device = True
    export_legacy = True

    @staticmethod
    def forward(
        Nx: Int[torch.Tensor, ""],
        Ny: Int[torch.Tensor, ""],
        dtype: torch.dtype,
        device: torch.device,
    ) -> Float[torch.Tensor, "N 2"]:
        # note: extra .to(dtype=) seems required for onnx export dtype correctness
        # seems like there is some dependence on torch default dtype inside linspace even when dtype argument is provided

        X = torch.linspace(-1.0, 1.0, Nx, dtype=dtype, device=device).to(dtype=dtype)
        Y = torch.linspace(-1.0, 1.0, Ny, dtype=dtype, device=device).to(dtype=dtype)
        Xgrid, Ygrid = torch.meshgrid(X, Y, indexing="xy")
        return torch.stack((Xgrid.reshape(-1), Ygrid.reshape(-1)), dim=-1)

    @staticmethod
    def example_inputs(
        dtype: torch.dtype, device: torch.device
    ) -> tuple[torch.Tensor, ...]:
        return tuple()

    @staticmethod
    def example_params(
        dtype: torch.dtype, device: torch.device
    ) -> tuple[torch.Tensor, ...]:
        return (
            torch.tensor(10, dtype=torch.int64),
            torch.tensor(11, dtype=torch.int64),
        )


class DiskSampling2DKernel(FunctionalKernel):
    input_names = []
    param_names = ["Nrho", "Ntheta"]
    output_names = ["samples"]
    forward_dtype_device = True
    export_legacy = True

    @staticmethod
    def forward(
        Nrho: Int[torch.Tensor, ""],
        Ntheta: Int[torch.Tensor, ""],
        dtype: torch.dtype,
        device: torch.device,
    ) -> Float[torch.Tensor, "N 2"]:
        samples = disk_sampling(Nrho, Ntheta, dtype, device)
        assert samples.dtype == dtype
        return samples

    @staticmethod
    def example_inputs(
        dtype: torch.dtype, device: torch.device
    ) -> tuple[torch.Tensor, ...]:
        return tuple()

    @staticmethod
    def example_params(
        dtype: torch.dtype, device: torch.device
    ) -> tuple[torch.Tensor, ...]:
        return (
            torch.tensor(10, dtype=torch.int64),
            torch.tensor(11, dtype=torch.int64),
        )
