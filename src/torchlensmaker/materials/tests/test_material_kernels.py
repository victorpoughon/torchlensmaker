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

import pytest

import torch

from torchlensmaker.materials.material_kernels import (
    NonDispersiveMaterialKernel,
    CauchyMaterialKernel,
    SellmeierMaterialKernel,
)

from torchlensmaker.core.functional_kernel import FunctionalKernel

from torchlensmaker.testing.test_functional_kernels_testing import (
    check_kernels_example_inputs_and_params,
    check_kernels_eval,
    check_kernels_export_onnx,
)

kernels_library: Dict[str, FunctionalKernel] = {
    "NonDispersiveMaterial": NonDispersiveMaterialKernel(),
    "CauchyMaterial": CauchyMaterialKernel(),
    "SellmeirMaterial": SellmeierMaterialKernel(),
}


def test_kinematics_kernels_inputs_and_params(
    dtype: torch.dtype, device: torch.device
) -> None:
    # Export, load, compare eval on example inputs
    for name, kernel in kernels_library.items():
        check_kernels_example_inputs_and_params(name, kernel, dtype, device)


def test_kinematics_kernels_eval(dtype: torch.dtype, device: torch.device) -> None:
    # Export, load, compare eval on example inputs
    for name, kernel in kernels_library.items():
        check_kernels_eval(name, kernel, dtype, device)


def test_kinematics_kernels_export_onnx(
    dtype: torch.dtype, device: torch.device, tmp_path: Path
) -> None:
    # Export, load, compare eval on example inputs
    for name, kernel in kernels_library.items():
        check_kernels_export_onnx(name, kernel, tmp_path, dtype, device)
