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

from typing import Type
from itertools import chain
from dataclasses import dataclass, is_dataclass, fields, astuple
from torchlensmaker.types import Tf2D, Tf3D
from typing import List, Any, cast, TypeAlias

KernelIOType: TypeAlias = torch.Tensor | Tf2D | Tf3D


torch.export.register_dataclass(Tf2D)
torch.export.register_dataclass(Tf3D)

def kernel_names(args: list[tuple[str, Type[KernelIOType]]]) -> list[str]:
    return [name for name, typ in args.items()]

def kernel_flat_io(
    t: KernelIOType | tuple[KernelIOType, ...],
) -> tuple[torch.Tensor, ...]:
    "Transform a tree of kernel inputs or outputs into a flat tupl of tensors"

    if isinstance(t, torch.Tensor):
        return (t,)
    if is_dataclass(t):
        return astuple(t)
    if isinstance(t, tuple):
        return tuple(e for elem in t for e in kernel_flat_io(elem))
    else:
        raise RuntimeError("flatten_kernel_outputs(): Invalid kernel output type")


def kernel_flat_names(args: dict[str, Type[KernelIOType]]) -> list[str]:
    flat_names: list[str] = []
    for name, typ in args.items():
        if is_dataclass(typ):
            flat_names.extend([name + "." + f.name for f in fields(typ)])
        else:
            flat_names.append(name)

    return flat_names


class FunctionalKernel:
    inputs: dict[str, Type[KernelIOType]]
    params: dict[str, Type[KernelIOType]]
    outputs: dict[str, Type[KernelIOType]]
    forward_dtype_device: bool = (
        False  # true if the kernel forward() function takes dtype and device arguments
    )
    export_legacy: bool = False  # true if onnx export must use legacy torch script (instead of default dynamo)

    @staticmethod
    def forward(*args: Any) -> KernelIOType | tuple[KernelIOType, ...]:
        raise NotImplementedError

    @staticmethod
    def example_inputs(
        dtype: torch.dtype, device: torch.device
    ) -> tuple[KernelIOType, ...]:
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

    flat_input_names = kernel_flat_names(kernel.inputs)
    flat_param_names = kernel_flat_names(kernel.params)
    flat_output_names = kernel_flat_names(kernel.outputs)

    onnx_program = cast(
        ONNXProgram,
        torch.onnx.export(
            FuncModule(kernel_forward),
            example_inputs,
            input_names=flat_input_names + flat_param_names,
            output_names=flat_output_names,
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

    flat_input_names = kernel_flat_names(kernel.inputs)
    flat_param_names = kernel_flat_names(kernel.params)
    flat_output_names = kernel_flat_names(kernel.outputs)

    torch.onnx.export(
        FuncModule(kernel_forward),
        example_inputs,
        input_names=flat_input_names + flat_param_names,
        output_names=flat_output_names,
        f=model_path,
        opset_version=18,
        dynamo=False,
    )
