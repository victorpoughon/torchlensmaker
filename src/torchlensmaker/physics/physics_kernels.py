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
from torchlensmaker.types import (
    ScalarTensor,
    BatchNDTensor,
    BatchTensor,
)
from torchlensmaker.core.functional_kernel import FunctionalKernel
from .physics import reflection, refraction


class ReflectionKernel(FunctionalKernel):
    inputs = {"rays": BatchNDTensor, "normals": BatchNDTensor}
    params = {}
    outputs = {"reflected": BatchNDTensor}

    @staticmethod
    def forward(rays: BatchNDTensor, normals: BatchNDTensor) -> BatchNDTensor:
        return reflection(rays, normals)

    @staticmethod
    def example_inputs(
        dtype: torch.dtype, device: torch.device
    ) -> tuple[BatchNDTensor, BatchNDTensor]:
        return (
            torch.tensor([[0.0, 1.0]], dtype=dtype, device=device),
            torch.tensor([[0.0, -1.0]], dtype=dtype, device=device),
        )

    @staticmethod
    def example_params(
        dtype: torch.dtype, device: torch.device
    ) -> tuple[torch.Tensor, ...]:
        return tuple()
