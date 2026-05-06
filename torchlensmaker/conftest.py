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

import pytest
import torch


def _normalize_device(device_str: str) -> torch.device:
    device_str = device_str.lower()
    if device_str == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    if device_str == "cpu":
        return torch.device("cpu")
    return torch.device(device_str)


def _normalize_dtype(dtype_str: str) -> torch.dtype:
    dtype_str = dtype_str.lower()
    if dtype_str == "float32" or dtype_str == "fp32":
        return torch.float32
    if dtype_str == "float64" or dtype_str == "fp64":
        return torch.float64
    raise ValueError(f"Unsupported dtype: {dtype_str}")


def pytest_addoption(parser):
    parser.addoption(
        "--device",
        action="store",
        default="cpu",
        help="Default device for tests: cpu, cuda",
    )
    parser.addoption(
        "--dtype",
        action="store",
        default="float32",
        help="Default dtype for tests: float32, float64",
    )


def pytest_configure(config):
    device_str = config.getoption("--device")
    dtype_str = config.getoption("--dtype")

    device = _normalize_device(device_str)
    dtype = _normalize_dtype(dtype_str)

    torch.set_default_device(device)
    torch.set_default_dtype(dtype)

    print(f"PyTorch defaults set to: device={device}, dtype={dtype}")


@pytest.fixture(scope="session")
def device(request):
    device_str = request.config.getoption("--device")
    return _normalize_device(device_str)


@pytest.fixture(scope="session")
def dtype(request):
    dtype_str = request.config.getoption("--dtype")
    return _normalize_dtype(dtype_str)


@pytest.fixture(scope="session")
def onnx_output_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("onnx", numbered=False)
