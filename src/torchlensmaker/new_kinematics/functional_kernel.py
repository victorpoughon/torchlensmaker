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
from torch.onnx import ONNXProgram

from typing import List, Any, cast


class FunctionalKernel:
    input_names: List[str]
    param_names: List[str]
    output_names: List[str]

    @staticmethod
    def forward(*args: Any) -> Any:
        raise NotImplementedError

    @staticmethod
    def example_inputs(
        dtype: torch.dtype, device: torch.device
    ) -> tuple[torch.Tensor, ...]:
        raise NotImplementedError

    @staticmethod
    def example_params(
        dtype: torch.dtype, device: torch.device
    ) -> tuple[torch.Tensor, ...]:
        raise NotImplementedError


class FuncModule(nn.Module):
    """
    Wrap a pure function into a nn.Module
    Useful because torch.onnx.export only accepts nn.Module
    """

    def __init__(self, func: Any):
        super().__init__()
        self.func = func

    def forward(self, *args: Any) -> Any:
        return self.func(*args)


def export_onnx(
    kernel: FunctionalKernel, dtype: torch.dtype, device: torch.device
) -> ONNXProgram:
    example_inputs = (
        *kernel.example_inputs(dtype, device),
        *kernel.example_params(dtype, device),
    )

    return cast(
        ONNXProgram,
        torch.onnx.export(
            FuncModule(kernel.forward),
            example_inputs,
            dynamo=True,
            input_names=kernel.input_names + kernel.param_names,
            output_names=kernel.output_names,
        ),
    )
