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
from typing import Dict

import pytest
import torch

from torchlensmaker.core.functional_kernel import FunctionalKernel
from torchlensmaker.light_sources.source_geometry_kernels import (
    ObjectGeometry2DKernel,
    ObjectGeometry3DKernel,
)
from torchlensmaker.testing.test_functional_kernels_testing import (
    check_kernels_eval,
    check_kernels_example_inputs_and_params,
    check_kernels_export_onnx,
)

kernels_library: Dict[str, FunctionalKernel] = {
    "ObjectGeometry2D": ObjectGeometry2DKernel(),
    "ObjectGeometry3D": ObjectGeometry3DKernel(),
}


def test_source_type_kernels_inputs_and_params(
    dtype: torch.dtype, device: torch.device
) -> None:
    # Export, load, compare eval on example inputs
    for name, kernel in kernels_library.items():
        check_kernels_example_inputs_and_params(name, kernel, dtype, device)


def test_source_type_kernels_eval(dtype: torch.dtype, device: torch.device) -> None:
    # Export, load, compare eval on example inputs
    for name, kernel in kernels_library.items():
        check_kernels_eval(name, kernel, dtype, device)


def test_source_type_kernels_export_onnx(
    dtype: torch.dtype, device: torch.device, tmp_path: Path
) -> None:
    # Note this test only works in float32 as of Jan 2026
    # because onnxruntime cpu doesn't seem to support cos() in float64...
    if dtype == torch.float64:
        return

    # Export, load, compare eval on example inputs
    for name, kernel in kernels_library.items():
        check_kernels_export_onnx(name, kernel, tmp_path, dtype, device)
