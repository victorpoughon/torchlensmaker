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


from dataclasses import astuple, fields, is_dataclass
from typing import Any, Type, TypeAlias, cast

import torch
import torch.nn as nn
from torch.onnx import ONNXProgram

from torchlensmaker.types import Tf

KernelIOType: TypeAlias = torch.Tensor | Tf


torch.export.register_dataclass(Tf)


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
            flat_names.extend([f"{name}.{field.name}" for field in fields(typ)])
        else:
            flat_names.append(name)

    return flat_names


class FunctionalKernel:
    inputs: dict[str, Any]
    params: dict[str, Any]
    outputs: dict[str, Any]
    dynamic_shapes: dict[str, Any] | None = None
    forward_dtype_device: bool = (
        False  # true if the kernel forward() function takes dtype and device arguments
    )
    export_legacy: bool = False  # true if onnx export must use legacy torch script (instead of default dynamo)
    export_onnx: bool = True  # False to disable onnx export entirely

    def example_inputs(
        self, dtype: torch.dtype, device: torch.device
    ) -> tuple[KernelIOType, ...]:
        raise NotImplementedError

    def example_params(
        self, dtype: torch.dtype, device: torch.device
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


def dynamo_dynamic_shapes(
    names: list[str], dynamic_shapes: dict[str, Any]
) -> dict[str, Any]:
    "Prepare kernel dynamic shapes into the argument to the dynamo export function"

    # Must have all input entries in the correct order
    ret = {}
    for name in names:
        if name in dynamic_shapes:
            ret[name] = dynamic_shapes[name]
        else:
            ret[name] = {}

    # Kernel inputs are given as positional args,
    # so we need to wrap dynamic_shapes in a dict with the "args" key
    # this is not well documented in pytorch
    return {"args": tuple(ret.values())}


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
        return kernel.apply(*args, dtype=dtype, device=device)

    kernel_forward = (
        forward_with_bound_dtype_device if kernel.forward_dtype_device else kernel.apply
    )

    flat_input_names = kernel_flat_names(kernel.inputs)
    flat_param_names = kernel_flat_names(kernel.params)
    flat_output_names = kernel_flat_names(kernel.outputs)

    if kernel.dynamic_shapes is not None:
        dynamic_shapes = dynamo_dynamic_shapes(
            flat_input_names + flat_param_names, kernel.dynamic_shapes
        )
    else:
        dynamic_shapes = None

    onnx_program = cast(
        ONNXProgram,
        torch.onnx.export(
            FuncModule(kernel_forward),
            example_inputs,
            input_names=flat_input_names + flat_param_names,
            output_names=flat_output_names,
            dynamic_shapes=dynamic_shapes,
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
        return kernel.apply(*args, dtype=dtype, device=device)

    kernel_forward = (
        forward_with_bound_dtype_device if kernel.forward_dtype_device else kernel.apply
    )

    flat_input_names = kernel_flat_names(kernel.inputs)
    flat_param_names = kernel_flat_names(kernel.params)
    flat_output_names = kernel_flat_names(kernel.outputs)

    # TODO support dynamic_axes in legacy export

    torch.onnx.export(
        FuncModule(kernel_forward),
        example_inputs,
        input_names=flat_input_names + flat_param_names,
        output_names=flat_output_names,
        f=model_path,
        opset_version=18,
        dynamo=False,
    )
