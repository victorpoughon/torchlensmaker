import pytest

from typing import Any

import torch
import torch.nn as nn

import torchlensmaker as tlm

"""
Test all surfaces using the common base class LocalSurface() methods,
local_collide() is tested only roughly, a more detailed test is in test_local_collide
"""

@pytest.fixture(params=[torch.float32, torch.float64], ids=["float32", "float64"])
def dtype(request: pytest.FixtureRequest) -> Any:
    return request.param

@pytest.fixture
def surfaces(dtype: torch.dtype,) -> list[tlm.LocalSurface]:
    # fmt: off
    return [
        tlm.Sphere(diameter=5, R=10, dtype=dtype),
        tlm.Sphere(diameter=5, C=0.05, dtype=dtype),
        tlm.Sphere(diameter=5, R=tlm.parameter(10), dtype=dtype),
        tlm.Sphere(diameter=5, C=tlm.parameter(10), dtype=dtype),

        tlm.SphereR(diameter=5, R=10, dtype=dtype),
        tlm.SphereR(diameter=5, C=0.05, dtype=dtype),
        tlm.SphereR(diameter=5, R=tlm.parameter(10), dtype=dtype),
        tlm.SphereR(diameter=5, C=tlm.parameter(10), dtype=dtype),

        tlm.Parabola(diameter=5, a=0.05, dtype=dtype),
        tlm.Parabola(diameter=5, a=tlm.parameter(0.05), dtype=dtype),
    ]
    # fmt: on


def test_testname(surfaces: list[tlm.LocalSurface]) -> None:
    for s in surfaces:
        assert len(s.testname()) > 0


def test_parameters(surfaces: list[tlm.LocalSurface]) -> None:
    for s in surfaces:
        params = s.parameters()
        assert isinstance(params, dict)
        assert all([isinstance(key, str) for key in params.keys()])
        assert all([isinstance(value, nn.Parameter) for value in params.values()])

def test_extent(surfaces: list[tlm.LocalSurface]) -> None:
    # isfinite
    # has surface dtype
    # has shape (dim,)
    ...


def test_extent_x(surfaces: list[tlm.LocalSurface]) -> None:
    # isfinite
    # is non zero except for plane and flat sphere
    # has surface dtype
    # has dim 0
    ...

def test_contains(surfaces: list[tlm.LocalSurface]) -> None:
    # test contains(samples) is true everywhere

    ...

# TODO rename samples functions and add to LocalSurface base class

# normals are finite everywhere and unit vectors
# samples are finite

# local collide basic stuff: shape, dim, batch shapes, isfinite




# for test_implicit_surface
# - F and F grad should be finite everywhere
# - F should be zero on samples
# - F should be non zero outside of bounding sphere/box
# - batch shapes of F and F_grad (all dimensions except last are preserved batch dims)
