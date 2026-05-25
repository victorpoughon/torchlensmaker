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
from jaxtyping import Float, Int

from torchlensmaker.core.functional_kernel import FunctionalKernel
from torchlensmaker.types import Batch2DTensor, BatchTensor

from .sampling import disk_sampling


class ZeroSampling1DKernel(FunctionalKernel):
    inputs = {}
    params = {}
    outputs = {"samples": BatchTensor}
    forward_dtype_device = True

    def apply(
        self, dtype: torch.dtype, device: torch.device
    ) -> Float[torch.Tensor, " 1"]:
        return torch.zeros((1), dtype=dtype, device=device)

    def example_inputs(
        self, dtype: torch.dtype, device: torch.device
    ) -> tuple[torch.Tensor, ...]:
        return tuple()

    def example_params(
        self, dtype: torch.dtype, device: torch.device
    ) -> tuple[torch.Tensor, ...]:
        return tuple()


class ZeroSampling2DKernel(FunctionalKernel):
    inputs = {}
    params = {}
    outputs = {"samples": Batch2DTensor}
    forward_dtype_device = True

    def apply(
        self, dtype: torch.dtype, device: torch.device
    ) -> Float[torch.Tensor, "1 2"]:
        return torch.zeros((1, 2), dtype=dtype, device=device)

    def example_inputs(
        self, dtype: torch.dtype, device: torch.device
    ) -> tuple[torch.Tensor, ...]:
        return tuple()

    def example_params(
        self, dtype: torch.dtype, device: torch.device
    ) -> tuple[torch.Tensor, ...]:
        return tuple()


class ExactSampling1DKernel(FunctionalKernel):
    inputs = {}
    params = {"ref_samples": BatchTensor}
    outputs = {"samples": BatchTensor}
    # https://github.com/pytorch/pytorch/issues/173076
    export_legacy = True
    forward_dtype_device = True

    def apply(
        self,
        ref_samples: Float[torch.Tensor, " N"],
        dtype: torch.dtype,
        device: torch.device,
    ) -> Float[torch.Tensor, " N"]:
        return ref_samples.to(dtype=dtype, device=device)

    def example_inputs(
        self, dtype: torch.dtype, device: torch.device
    ) -> tuple[torch.Tensor, ...]:
        return tuple()

    def example_params(
        self, dtype: torch.dtype, device: torch.device
    ) -> tuple[torch.Tensor, ...]:
        return (torch.tensor([-0.5, 0.0, 0.5], dtype=dtype, device=device),)


class ExactSampling2DKernel(FunctionalKernel):
    inputs = {}
    params = {"ref_samples": Batch2DTensor}
    outputs = {"samples": Batch2DTensor}
    # https://github.com/pytorch/pytorch/issues/173076
    export_legacy = True
    forward_dtype_device = True

    def apply(
        self,
        ref_samples: Float[torch.Tensor, "N 2"],
        dtype: torch.dtype,
        device: torch.device,
    ) -> Float[torch.Tensor, "N 2"]:
        return ref_samples.to(dtype=dtype, device=device)

    def example_inputs(
        self, dtype: torch.dtype, device: torch.device
    ) -> tuple[torch.Tensor, ...]:
        return tuple()

    def example_params(
        self, dtype: torch.dtype, device: torch.device
    ) -> tuple[torch.Tensor, ...]:
        return (
            torch.tensor(
                [
                    [0, 0],
                    [-1, 0],
                    [0, -1],
                    [-1, -1],
                    [0, 1],
                    [1, 0],
                    [1, 1],
                    [-1, 1],
                    [1, -1],
                ],
                dtype=dtype,
                device=device,
            ),
        )


def _linspace_centered(
    N: Int[torch.Tensor, ""], dtype: torch.dtype, device: torch.device
) -> Float[torch.Tensor, " N"]:
    "linspace from -1 to 1 with N samples; returns [0.0] when N==1 instead of [-1.0]"
    # note: this is not quite mergeable in main
    # due to onnx export issues with if N == 1
    if N == 1:
        return torch.zeros(1, dtype=dtype, device=device)
    one = torch.ones((), dtype=dtype, device=device)
    # note: extra .to(dtype=) seems required for onnx export dtype correctness
    # seems like there is some dependence on torch default dtype inside linspace even when dtype argument is provided
    return torch.linspace(-one, one, N, dtype=dtype, device=device).to(dtype=dtype)


class LinspaceSampling1DKernel(FunctionalKernel):
    inputs = {}
    params = {"N": Int[torch.Tensor, ""]}
    outputs = {"samples": BatchTensor}
    forward_dtype_device = True
    export_legacy = True

    def apply(
        self, N: Int[torch.Tensor, ""], dtype: torch.dtype, device: torch.device
    ) -> Float[torch.Tensor, " N"]:
        return _linspace_centered(N, dtype, device)

    def example_inputs(
        self, dtype: torch.dtype, device: torch.device
    ) -> tuple[torch.Tensor, ...]:
        return tuple()

    def example_params(
        self, dtype: torch.dtype, device: torch.device
    ) -> tuple[torch.Tensor, ...]:
        return (torch.tensor(10, dtype=torch.int64, device=device),)


class LinspaceSampling2DKernel(FunctionalKernel):
    inputs = {}
    params = {"Nx": Int[torch.Tensor, ""], "Ny": Int[torch.Tensor, ""]}
    outputs = {"samples": Batch2DTensor}
    forward_dtype_device = True
    export_legacy = True

    def apply(
        self,
        Nx: Int[torch.Tensor, ""],
        Ny: Int[torch.Tensor, ""],
        dtype: torch.dtype,
        device: torch.device,
    ) -> Float[torch.Tensor, "N 2"]:
        X = _linspace_centered(Nx, dtype, device)
        Y = _linspace_centered(Ny, dtype, device)
        Xgrid, Ygrid = torch.meshgrid(X, Y, indexing="xy")
        return torch.stack((Xgrid.reshape(-1), Ygrid.reshape(-1)), dim=-1)

    def example_inputs(
        self, dtype: torch.dtype, device: torch.device
    ) -> tuple[torch.Tensor, ...]:
        return tuple()

    def example_params(
        self, dtype: torch.dtype, device: torch.device
    ) -> tuple[torch.Tensor, ...]:
        return (
            torch.tensor(10, dtype=torch.int64, device=device),
            torch.tensor(11, dtype=torch.int64, device=device),
        )


class DiskSampling2DKernel(FunctionalKernel):
    inputs = {}
    params = {"Nrho": Int[torch.Tensor, ""], "Ntheta": Int[torch.Tensor, ""]}
    outputs = {"samples": Batch2DTensor}
    forward_dtype_device = True
    export_legacy = True

    def apply(
        self,
        Nrho: Int[torch.Tensor, ""],
        Ntheta: Int[torch.Tensor, ""],
        dtype: torch.dtype,
        device: torch.device,
    ) -> Float[torch.Tensor, "N 2"]:
        samples = disk_sampling(Nrho, Ntheta, dtype, device)
        assert samples.dtype == dtype
        return samples

    def example_inputs(
        self, dtype: torch.dtype, device: torch.device
    ) -> tuple[torch.Tensor, ...]:
        return tuple()

    def example_params(
        self, dtype: torch.dtype, device: torch.device
    ) -> tuple[torch.Tensor, ...]:
        return (
            torch.tensor(10, dtype=torch.int64, device=device),
            torch.tensor(11, dtype=torch.int64, device=device),
        )
