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

from pathlib import Path
from itertools import chain

import itertools
from typing import TypeAlias
from dataclasses import is_dataclass
import pytest

import torch
import onnxruntime

from torchlensmaker.types import Tf2D, Tf3D
from torchlensmaker.core.functional_kernel import (
    export_onnx,
    FunctionalKernel,
    KernelIOType,
    kernel_flat_io,
    kernel_flat_names,
    kernel_names,
)


def check_kernels_example_inputs_and_params(
    name: str, kernel: FunctionalKernel, dtype: torch.dtype, device: torch.device
) -> None:
    example_inputs = kernel.example_inputs(dtype, device)
    assert isinstance(example_inputs, tuple)
    assert len(example_inputs) == len(kernel.inputs)
    assert all(t.dtype == dtype for t in example_inputs)

    example_params = kernel.example_params(dtype, device)
    assert isinstance(example_params, tuple)
    assert len(example_params) == len(kernel.params)
    # dont check dtype of params because it can be different than the main dtype
    # in some cases, e.g. sampling

    expected_num_outputs = len(kernel.outputs)
    assert expected_num_outputs != 0, "Kernel with no output is not supported"

    # Verify no duplicate names
    all_names = (
        kernel_names(kernel.inputs)
        + kernel_names(kernel.params)
        + kernel_names(kernel.outputs)
    )
    assert len(all_names) == len(set(all_names))


def check_kernels_eval(
    name: str,
    kernel: FunctionalKernel,
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    # Evaluate kernel with example inputs
    kwargs = dict(dtype=dtype, device=device) if kernel.forward_dtype_device else {}
    kernel_outputs = kernel.forward(
        *kernel.example_inputs(dtype, device),
        *kernel.example_params(dtype, device),
        **kwargs,
    )

    assert isinstance(kernel_outputs, (tuple, KernelIOType)), (
        f"Kernel return type not allowed: {type(kernel_outputs)}"
    )

    # Check number of outputs
    expected_num_outputs = len(kernel.outputs)
    if expected_num_outputs == 1:
        assert not isinstance(kernel_outputs, tuple), (
            "Kernel with a single output must not return a tuple"
        )
    else:
        assert isinstance(kernel_outputs, tuple), (
            "Kernel with multiple outputs must return a tuple"
        )
        assert expected_num_outputs == len(kernel_outputs)

    # Check dtype and device
    for actual in kernel_flat_io(kernel_outputs):
        # either bool for masks, or the input dtype
        assert actual.dtype == torch.bool or actual.dtype == dtype, (
            f"Expected kernel output dtype {dtype}, got {actual.dtype}"
        )
        assert actual.device == device, (
            f"Expected kernel output dtype {device}, got {actual.device}"
        )


def check_kernels_export_onnx(
    name: str,
    kernel: FunctionalKernel,
    tmp_path: Path,
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    # Export model to ONNX file
    model_path = tmp_path / f"{name}.onnx"
    export_onnx(model_path, kernel, dtype, device)

    # Load the exported model
    onnx_inputs = [
        t.numpy(force=True)
        for t in kernel_flat_io(kernel.example_inputs(dtype, device))
    ]
    onnx_params = [t.numpy(force=True) for t in kernel.example_params(dtype, device)]

    ort_session = onnxruntime.InferenceSession(
        model_path, providers=["CPUExecutionProvider"]
    )

    # Check that inputs, params and outputs names match
    model_input_names = [inpt.name for inpt in ort_session.get_inputs()]
    model_output_names = [output.name for output in ort_session.get_outputs()]
    flat_input_names = kernel_flat_names(kernel.inputs)
    flat_param_names = kernel_flat_names(kernel.params)
    flat_output_names = kernel_flat_names(kernel.outputs)

    assert model_input_names == flat_input_names + flat_param_names
    assert model_output_names == flat_output_names

    # Evaluate both on example inputs
    ort_input = {
        input_arg.name: input_value
        for input_arg, input_value in zip(
            ort_session.get_inputs(), chain(onnx_inputs, onnx_params)
        )
    }

    kwargs = dict(dtype=dtype, device=device) if kernel.forward_dtype_device else {}

    ort_outputs = ort_session.run(None, ort_input)
    kernel_outputs_tensors = kernel_flat_io(
        kernel.forward(
            *kernel.example_inputs(dtype, device),
            *kernel.example_params(dtype, device),
            **kwargs,
        )
    )

    kernel_outputs = [t.numpy(force=True) for t in kernel_outputs_tensors]

    # Compare values, dtype, shape
    for actual, expected, arg in zip(ort_outputs, kernel_outputs, kernel.outputs.items()):
        assert actual.dtype == expected.dtype, (actual.dtype, expected.dtype)
        assert actual.shape == expected.shape, arg
        torch.testing.assert_close(actual, expected)
