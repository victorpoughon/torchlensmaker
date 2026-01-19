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
    forward_dtype_device: bool = False # true if the kernel forward() function takes dtype and device arguments
    export_legacy: bool = False # true if onnx export must use legacy torch script (instead of default dynamo)

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
    model_path: str, kernel: FunctionalKernel, dtype: torch.dtype, device: torch.device
) -> None:
    if kernel.export_legacy:
        export_onnx_legacy(model_path, kernel, dtype, device)
    else:
        export_onnx_dynamo(model_path, kernel, dtype, device)


def export_onnx_dynamo(
    model_path: str, kernel: FunctionalKernel, dtype: torch.dtype, device: torch.device
) -> None:
    """
    Export a functional kernel to a ONNX file
    """

    example_inputs = (
        *kernel.example_inputs(dtype, device),
        *kernel.example_params(dtype, device),
    )

    def forward_with_bound_dtype_device(*args):
        return kernel.forward(*args, dtype=dtype, device=device)

    kernel_forward = (
        forward_with_bound_dtype_device
        if kernel.forward_dtype_device
        else kernel.forward
    )

    onnx_program = cast(
        ONNXProgram,
        torch.onnx.export(
            FuncModule(kernel_forward),
            example_inputs,
            input_names=kernel.input_names + kernel.param_names,
            output_names=kernel.output_names,
            opset_version=18,
            dynamo=True,
        ),
    )

    onnx_program.save(model_path)


def export_onnx_legacy(
    model_path: str, kernel: FunctionalKernel, dtype: torch.dtype, device: torch.device
) -> None:
    """
    Export a functional kernel to a ONNX file
    """

    example_inputs = (
        *kernel.example_inputs(dtype, device),
        *kernel.example_params(dtype, device),
    )

    def forward_with_bound_dtype_device(*args):
        return kernel.forward(*args, dtype=dtype, device=device)

    kernel_forward = (
        forward_with_bound_dtype_device
        if kernel.forward_dtype_device
        else kernel.forward
    )

    torch.onnx.export(
        FuncModule(kernel_forward),
        example_inputs,
        input_names=kernel.input_names + kernel.param_names,
        output_names=kernel.output_names,
        f=model_path,
        opset_version=18,
        dynamo=False,
    )
