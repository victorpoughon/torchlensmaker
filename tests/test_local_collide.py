import pytest
from dataclasses import dataclass

import torchlensmaker as tlm
import torch

from torchlensmaker.testing.collision_datasets import (
    NormalRays,
    FixedRays,
    RayGenerator,
    OrbitalRays,
)

from torchlensmaker.core.geometry import unit2d_rot, unit3d_rot

from torchlensmaker.testing.check_local_collide import check_local_collide
from torchlensmaker.testing.collision_datasets import make_offset_rays

from typing import Any

from .conftest import make_common_surfaces

# TODO
# Rays from normal generator and returned normals should be parallel, check dot product is close to one
# assert torch.allclose(
#     torch.sum(V * local_normals, dim=-1),
#     torch.ones(V.shape[:-1], dtype=V.dtype),
# )


@dataclass
class CollisionTestCase:
    generator: list[RayGenerator]
    expected_collide: bool


def make_test_cases_all_surfaces() -> list[CollisionTestCase]:
    "List of test cases that are common to all surfaces"

    generators_collide: list[RayGenerator] = []
    generators_no_collide: list[RayGenerator] = []
    N = 12
    epsilon = 0.05

    for dim in (2, 3):
        # fmt: off
        # GENERATORS EXPECTED TO COLLIDE
        generators_collide.extend([
            NormalRays(dim=dim, N=N, offset=0.0, epsilon=epsilon),
        ])

        if dim == 2:
            generators_collide.extend([
                # TODO rays with V_x == 0 are out of scope for now
                # FixedRays(direction=torch.tensor([0.0, 1.0]), dim=dim, N=N, offset=0.0, epsilon=epsilon),
                FixedRays(direction=torch.tensor([1.0, 0.0]), dim=dim, N=N, offset=0.0, epsilon=epsilon),
                FixedRays(direction=unit2d_rot(110), dim=dim, N=N, offset=0.0, epsilon=epsilon),
                FixedRays(direction=unit2d_rot(90), dim=dim, N=N, offset=0.0, epsilon=epsilon),
                FixedRays(direction=unit2d_rot(80), dim=dim, N=N, offset=0.0, epsilon=epsilon),
                FixedRays(direction=unit2d_rot(60), dim=dim, N=N, offset=0.0, epsilon=epsilon),
                FixedRays(direction=unit2d_rot(40), dim=dim, N=N, offset=0.0, epsilon=epsilon),
                FixedRays(direction=unit2d_rot(20), dim=dim, N=N, offset=0.0, epsilon=epsilon),
                FixedRays(direction=unit2d_rot(0), dim=dim, N=N, offset=0.0, epsilon=epsilon),
                FixedRays(direction=unit2d_rot(-10), dim=dim, N=N, offset=0.0, epsilon=epsilon),
            ])
        
        if dim == 3:
            generators_collide.extend([
                # FixedRays(direction=torch.tensor([0.0, 0.0, 1.0]), dim=dim, N=N, offset=0.0, epsilon=epsilon),
                # FixedRays(direction=torch.tensor([0.0, 1.0, 0.0]), dim=dim, N=N, offset=0.0, epsilon=epsilon),
                FixedRays(direction=torch.tensor([1.0, 0.0, 0.0]), dim=dim, N=N, offset=0.0, epsilon=epsilon),
                FixedRays(direction=unit3d_rot(110, 50), dim=dim, N=N, offset=0.0, epsilon=epsilon),
                FixedRays(direction=unit3d_rot(90, 20), dim=dim, N=N, offset=0.0, epsilon=epsilon),
                FixedRays(direction=unit3d_rot(80, -80), dim=dim, N=N, offset=0.0, epsilon=epsilon),
                FixedRays(direction=unit3d_rot(60, 80), dim=dim, N=N, offset=0.0, epsilon=epsilon),
                FixedRays(direction=unit3d_rot(40, 0), dim=dim, N=N, offset=0.0, epsilon=epsilon),
                FixedRays(direction=unit3d_rot(20, -40), dim=dim, N=N, offset=0.0, epsilon=epsilon),
                FixedRays(direction=unit3d_rot(0, 0), dim=dim, N=N, offset=0.0, epsilon=epsilon),
                FixedRays(direction=unit3d_rot(-10, -30), dim=dim, N=N, offset=0.0, epsilon=epsilon),
            ])
        
        # GENERATORS EXPECTED TO NOT COLLIDE
        if dim == 2:
            generators_no_collide.extend([
                    OrbitalRays(radius=1.1, dim=dim, N=N, offset=0., epsilon=0.),
                ])
        
        if dim == 3:
            # TODO no collide 3D
            generators_no_collide.extend([])

        # fmt: on

    return [CollisionTestCase(gen, True) for gen in generators_collide] + [
        CollisionTestCase(gen, False) for gen in generators_no_collide
    ]


@pytest.fixture(params=make_test_cases_all_surfaces())
def test_cases_all_surfaces(request: pytest.FixtureRequest) -> Any:
    return request.param


@pytest.fixture(
    params=make_common_surfaces(torch.float32) + make_common_surfaces(torch.float64)
)
def surface(request: pytest.FixtureRequest) -> Any:
    return request.param


def test_expected_collide(
    surface: tlm.LocalSurface, test_cases_all_surfaces: RayGenerator
) -> None:
    "Test cases common to all surfaces"

    offset_space = torch.cat(
        (
            torch.logspace(-6, 2, 5),
            torch.linspace(0.0, 50.0, 5),
            -torch.logspace(-6, 2, 5),
            -torch.linspace(0.0, 50.0, 5),
        ),
        dim=0,
    )

    genP, genV = test_cases_all_surfaces.generator(surface)

    # Add copies of rays but with origin points moved along the direction
    # this is to check the collision detection dependence on rays origins
    P, V = make_offset_rays(genP, genV, offset_space)

    print("Checking collision")
    print("Ray generator:", test_cases_all_surfaces.generator)
    print("Surface = ", surface.testname())
    print("expected collide:", test_cases_all_surfaces.expected_collide)
    print("dim", P.shape[1])
    print("dtype", surface.dtype)
    print()

    # Call local_collide and check
    check_local_collide(surface, P, V, expected_collide=test_cases_all_surfaces.expected_collide)
