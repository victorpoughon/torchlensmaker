import torch
import torchlensmaker as tlm

import pytest

from typing import Any

"""
Fixtures common to all tests
"""

def make_common_surfaces(dtype: torch.dtype) -> list[tlm.LocalSurface]:
    # fmt: off
    return [
        tlm.CircularPlane(diameter=30, dtype=dtype),

        # Non axially symmetric surfaces are out of scope for now
        # tlm.SquarePlane(side_length=30, dtype=dtype),

        # Sphere
        tlm.Sphere(diameter=5, R=10, dtype=dtype),
        tlm.Sphere(diameter=5, C=0.05, dtype=dtype),
        tlm.Sphere(diameter=5, C=0., dtype=dtype),
        tlm.Sphere(diameter=5, R=tlm.parameter(10, dtype=dtype), dtype=dtype),
        tlm.Sphere(diameter=5, C=tlm.parameter(0.05, dtype=dtype), dtype=dtype),

        # SphereR
        tlm.SphereR(diameter=5, R=10, dtype=dtype),
        tlm.SphereR(diameter=5, C=0.05, dtype=dtype),
        tlm.SphereR(diameter=5, R=tlm.parameter(10, dtype=dtype), dtype=dtype),
        tlm.SphereR(diameter=5, C=tlm.parameter(0.05, dtype=dtype), dtype=dtype),

        # Half circle with SphereR
        tlm.SphereR(diameter=5, R=2.5, dtype=dtype),
        tlm.SphereR(diameter=5, R=tlm.parameter(2.5, dtype=dtype), dtype=dtype),

        # Parabola
        tlm.Parabola(diameter=5, A=0.05, dtype=dtype),
        tlm.Parabola(diameter=5, A=tlm.parameter(0.05, dtype=dtype), dtype=dtype),
        tlm.Parabola(diameter=5, A=-0.05, dtype=dtype),
        tlm.Parabola(diameter=5, A=tlm.parameter(-0.05, dtype=dtype), dtype=dtype),
        tlm.Parabola(diameter=5, A=0, dtype=dtype),
        tlm.Parabola(diameter=5, A=tlm.parameter(0, dtype=dtype), dtype=dtype),

        # Asphere
        tlm.Asphere(diameter=10, R=50, K=1.0, A4=0.005, dtype=dtype),
        tlm.Asphere(diameter=10, R=-50, K=1.0, A4=0.005, dtype=dtype),
        tlm.Asphere(diameter=10, R=50, K=1.0, A4=0.005, dtype=dtype),
        tlm.Asphere(diameter=10, R=50, K=-1.0, A4=0.005, dtype=dtype),
        tlm.Asphere(diameter=10, R=50, K=1.0, A4=0.005, dtype=dtype),
        tlm.Asphere(diameter=10, R=50, K=1.0, A4=-0.005, dtype=dtype),
        tlm.Asphere(diameter=10, R=50, K=1.0, A4=0.005, dtype=dtype),
        tlm.Asphere(diameter=10, R=-50, K=1.0, A4=0.005, dtype=dtype),
        tlm.Asphere(diameter=10, R=50, K=1.0, A4=0.005, dtype=dtype),
        tlm.Asphere(diameter=10, R=50, K=-1.0, A4=0.005, dtype=dtype),
        tlm.Asphere(diameter=10, R=50, K=1.0, A4=0.005, dtype=dtype),
        tlm.Asphere(diameter=10, R=50, K=1.0, A4=-0.005, dtype=dtype),
        tlm.Asphere(diameter=10, R=tlm.parameter(50, dtype=dtype), K=1.0, A4=0.005, dtype=dtype),
        tlm.Asphere(diameter=10, R=-50, K=tlm.parameter(1.0, dtype=dtype), A4=0.005, dtype=dtype),
        tlm.Asphere(diameter=10, R=50, K=1.0, A4=tlm.parameter(0.005, dtype=dtype), dtype=dtype),

        # TODO test domains of partial surfaces?
    ]
    # fmt: on

@pytest.fixture
def surfaces(dtype: torch.dtype) -> list[tlm.LocalSurface]:
    return make_common_surfaces(dtype)

@pytest.fixture(params=[2, 3], ids=["2D", "3D"])
def dim(request: pytest.FixtureRequest) -> Any:
    return request.param


@pytest.fixture(params=[torch.float32, torch.float64], ids=["float32", "float64"])
def dtype(request: pytest.FixtureRequest) -> Any:
    return request.param
