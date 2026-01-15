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

from typing import Dict
from pathlib import Path
from itertools import chain

import pytest

import torch
import onnxruntime

from torchlensmaker.kinematics.kinematics_kernels import (
    Translate2DKernel,
    Translate3DKernel,
    Rotate2DKernel,
    Rotate3DKernel,
    AbsolutePosition3DKernel,
)

from torchlensmaker.kinematics.functional_kernel import export_onnx, FunctionalKernel


kernels_library: Dict[str, FunctionalKernel] = {
    "Rotate2D": Rotate2DKernel(),
    "Rotate3D": Rotate3DKernel(),
    "Translate2D": Translate2DKernel(),
    "Translate3D": Translate3DKernel(),
    "AbsolutePosition3D": AbsolutePosition3DKernel(),
}


def test_kernels_example_inputs_and_params() -> None:
    dtype = torch.float64
    device = torch.device("cpu")

    # returns tuple correct length
    # each tensor has correct shape / dtype / device

    for name, model in kernels_library.items():
        example_inputs = model.example_inputs(dtype, device)
        assert len(example_inputs) == len(model.input_names)
        assert all(t.dtype == dtype for t in example_inputs)
        assert all(t.device == device for t in example_inputs)

        example_params = model.example_params(dtype, device)
        assert len(example_params) == len(model.param_names)
        assert all(t.dtype == dtype for t in example_params)
        assert all(t.device == device for t in example_params)

        # Verify no duplicate names
        all_names = model.input_names + model.param_names + model.output_names
        assert len(all_names) == len(set(all_names))


def test_kernels_export_onnx(tmp_path: Path) -> None:
    # Note this test only works in float32 as of Jan 2026
    # because onnxruntime cpu doesn't seem to support cos() in float64...
    dtype = torch.float32
    device = torch.device("cpu")

    # Export, load, compare eval on example inputs
    for name, kernel in kernels_library.items():
        # Export model to ONNX file
        onnx_program = export_onnx(kernel, dtype, device)
        model_path = tmp_path / f"{name}.onnx"
        onnx_program.save(model_path)

        # Load the exported model
        onnx_inputs = [
            t.numpy(force=True) for t in kernel.example_inputs(dtype, device)
        ]
        onnx_params = [
            t.numpy(force=True) for t in kernel.example_params(dtype, device)
        ]

        ort_session = onnxruntime.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )

        # Check that inputs, params and outputs names match
        model_input_names = [inpt.name for inpt in ort_session.get_inputs()]
        model_output_names = [output.name for output in ort_session.get_outputs()]

        assert model_input_names == kernel.input_names + kernel.param_names
        assert model_output_names == kernel.output_names

        # Evaluate both on example inputs
        ort_input = {
            input_arg.name: input_value
            for input_arg, input_value in zip(
                ort_session.get_inputs(), chain(onnx_inputs, onnx_params)
            )
        }

        ort_outputs = ort_session.run(None, ort_input)
        kernel_outputs_tensors = kernel.forward(
            *kernel.example_inputs(dtype, device), *kernel.example_params(dtype, device)
        )
        kernel_outputs = [t.numpy(force=True) for t in kernel_outputs_tensors]

        # Compare values, dtype, shape
        for actual, expected in zip(ort_outputs, kernel_outputs):
            assert actual.dtype == expected.dtype
            assert actual.shape == expected.shape
            torch.testing.assert_close(actual, expected)
