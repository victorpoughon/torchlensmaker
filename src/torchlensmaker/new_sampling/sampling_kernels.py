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


class Linspace1DSampler(FunctionalKernel):
    input_names = ["nexus"]
    param_names = ["N"]
    output_names = ["samples"]

    @staticmethod
    def forward(
        nexus: torch.Tensor, N: Int[torch.Tensor, ""]
    ) -> Float[torch.Tensor, " N"]:
        one = torch.ones(1, dtype=nexus.dtype, device=nexus.device)
        return (torch.linspace(-one, one, N, dtype=nexus.dtype, device=nexus.device),)

    @staticmethod
    def example_inputs(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        return (torch.empty((), dtype=dtype, device=device),)

    @staticmethod
    def example_params(dtype: torch.dtype, device: torch.device) -> tuple[int]:
        return (torch.tensor(10, dtype=torch.int32),)
