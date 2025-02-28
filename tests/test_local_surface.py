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
        tlm.Sphere(diameter=5, C=0., dtype=dtype),
        tlm.Sphere(diameter=5, R=tlm.parameter(10, dtype=dtype), dtype=dtype),
        tlm.Sphere(diameter=5, C=tlm.parameter(0.05, dtype=dtype), dtype=dtype),

        tlm.SphereR(diameter=5, R=10, dtype=dtype),
        tlm.SphereR(diameter=5, C=0.05, dtype=dtype),
        tlm.SphereR(diameter=5, R=tlm.parameter(10, dtype=dtype), dtype=dtype),
        tlm.SphereR(diameter=5, C=tlm.parameter(0.05, dtype=dtype), dtype=dtype),

        tlm.Parabola(diameter=5, a=0.05, dtype=dtype),
        tlm.Parabola(diameter=5, a=tlm.parameter(0.05, dtype=dtype), dtype=dtype),
        tlm.Parabola(diameter=5, a=0., dtype=dtype),

        # TODO Asphere
        # TODO Planes
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


def isflat(s: tlm.LocalSurface) -> bool:
    return (
        isinstance(s, tlm.Plane)
        or (
            isinstance(s, tlm.Sphere)
            and torch.allclose(s.C, torch.tensor(0.0, dtype=s.dtype))
        )
        or (
            isinstance(s, tlm.Parabola)
            and torch.allclose(s.a, torch.tensor(0.0, dtype=s.dtype))
        )
    )


def test_extent_and_zero(surfaces: list[tlm.LocalSurface]) -> None:
    for s in surfaces:
        zero2 = s.zero(2)
        extent2 = s.extent(2)
        assert zero2.dim() == 1 and zero2.shape == (2,)
        assert extent2.dim() == 1 and extent2.shape == (2,)
        assert torch.all(torch.isfinite(zero2))
        assert torch.all(torch.isfinite(extent2))
        assert zero2.dtype == s.dtype
        assert extent2.dtype == s.dtype
        assert torch.allclose(zero2, torch.tensor(0., dtype=s.dtype))
        if not isflat(s):
            assert not torch.allclose(extent2, torch.tensor(0., dtype=s.dtype))
        else:
            assert torch.allclose(extent2, torch.tensor(0., dtype=s.dtype))
        del zero2, extent2

        zero3 = s.zero(3)
        extent3 = s.extent(3)
        assert zero3.dim() == 1 and zero3.shape == (3,)
        assert extent3.dim() == 1 and extent3.shape == (3,)
        assert torch.all(torch.isfinite(zero3))
        assert torch.all(torch.isfinite(extent3))
        assert zero3.dtype == s.dtype
        assert extent3.dtype == s.dtype
        if not isflat(s):
            assert not torch.allclose(extent3, torch.tensor(0., dtype=s.dtype))
        else:
            assert torch.allclose(extent3, torch.tensor(0., dtype=s.dtype))
        del zero3, extent3


    # is non zero except for plane and flat sphere


def test_normals(surfaces: list[tlm.LocalSurface]) -> None:
    ...

    # normals are finite everywhere and unit vectors


def test_contains_and_samples2D(surfaces: list[tlm.LocalSurface]) -> None:
    ...

    # samples2D_half
    # samples2D_full

    # finite
    # contain(samples) == true
    # contains(modified samples) == false

def test_local_collide_basic(surfaces: list[tlm.LocalSurface]) -> None:
    ...
    # local collide basic stuff: shape, dim, batch shapes, isfinite


# further for test_implicit_surface
# - F and F grad should be finite everywhere
# - F should be zero on samples
# - F should be non zero outside of bounding sphere/box
# - batch shapes of F and F_grad (all dimensions except last are preserved batch dims)
