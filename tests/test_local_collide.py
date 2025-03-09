# TODO port tests from notebook

# construction of test cases

# expected collide all surfaces:
# - normal
# - fixed direction
# - random direction

# expected no collide for all surfaces:
# - outside of bbox

# expected collide for some surfaces
# - tangent with fixed offset

# try multiple offset as part of check_collide?

# print test case before calling check_collide

import pytest

import torchlensmaker as tlm
import torch

from torchlensmaker.testing.collision_datasets import (
    NormalRays,
    FixedRays,
    RayGenerator,
)

from torchlensmaker.core.geometry import unit2d_rot, unit3d_rot

from torchlensmaker.testing.check_local_collide import check_local_collide
from torchlensmaker.testing.collision_datasets import make_offset_rays

from typing import Any

from .conftest import make_common_surfaces

# TODO
# Rays and returned normals should be parallel, check dot product is close to one
    # assert torch.allclose(
    #     torch.sum(V * local_normals, dim=-1),
    #     torch.ones(V.shape[:-1], dtype=V.dtype),
    # )

def expected_collide_all_surfaces() -> list[RayGenerator]:
    "Make a list of generators that are expected to collide for all surfaces"

    generators: list[RayGenerator] = []
    N = 12
    epsilon = 0.05

    for dim in (2, 3):
    
        generators.extend([
            NormalRays(dim=dim, N=N, offset=0.0, epsilon=epsilon),
        ])

        if dim == 2:
            generators.extend([
                #FixedRays(direction=torch.tensor([0.0, 1.0], dtype=dtype), dim=dim, N=N, offset=0.0, epsilon=epsilon),
                #FixedRays(direction=torch.tensor([1.0, 0.0], dtype=dtype), dim=dim, N=N, offset=0.0, epsilon=epsilon),
                FixedRays(direction=unit2d_rot(100), dim=dim, N=N, offset=0.0, epsilon=epsilon),
                #FixedRays(direction=unit2d_rot(90), dim=dim, N=N, offset=0.0, epsilon=epsilon),
                FixedRays(direction=unit2d_rot(80), dim=dim, N=N, offset=0.0, epsilon=epsilon),
                FixedRays(direction=unit2d_rot(70), dim=dim, N=N, offset=0.0, epsilon=epsilon),
                #FixedRays(direction=unit2d_rot(60), dim=dim, N=N, offset=0.0, epsilon=epsilon),
                FixedRays(direction=unit2d_rot(50), dim=dim, N=N, offset=0.0, epsilon=epsilon),
                #FixedRays(direction=unit2d_rot(40), dim=dim, N=N, offset=0.0, epsilon=epsilon),
                FixedRays(direction=unit2d_rot(30), dim=dim, N=N, offset=0.0, epsilon=epsilon),
                #FixedRays(direction=unit2d_rot(20), dim=dim, N=N, offset=0.0, epsilon=epsilon),
                FixedRays(direction=unit2d_rot(10), dim=dim, N=N, offset=0.0, epsilon=epsilon),
                FixedRays(direction=unit2d_rot(0), dim=dim, N=N, offset=0.0, epsilon=epsilon),
                FixedRays(direction=unit2d_rot(-10), dim=dim, N=N, offset=0.0, epsilon=epsilon),
            ])
        
        if dim == 3:
            generators.extend([])
        
    
    return generators


@pytest.fixture(params=expected_collide_all_surfaces())
def generator_expected_collide(request: pytest.FixtureRequest) -> Any:
    return request.param

@pytest.fixture(params=make_common_surfaces(torch.float32) + make_common_surfaces(torch.float64))
def surface(request: pytest.FixtureRequest) -> Any:
    return request.param


def test_expected_collide(surface: tlm.LocalSurface, generator_expected_collide: RayGenerator) -> None:
    "Test with rays that are expected to collide with all surfaces"

    offset_space = torch.cat(
        (
            torch.logspace(-6, 2, 5),
            torch.linspace(0.0, 50.0, 5),
            -torch.logspace(-6, 2, 5),
            -torch.linspace(0.0, 50.0, 5),
        ),
        dim=0,
    )

    gen = generator_expected_collide
    genP, genV = gen(surface)
    
    # Add copies of rays but with origin points moved along the direction
    # this is to check the collision detection dependence on rays origins
    P, V = make_offset_rays(genP, genV, offset_space)

    print("Checking collision")
    print("Ray generator:", gen)
    print("Surface = ", surface.testname())
    print("Expected collide = yes")
    print("dim", P.shape[1])
    print("dtype", surface.dtype)
    print()

    # Call local_collide and check
    check_local_collide(surface, P, V, expected_collide=True)
