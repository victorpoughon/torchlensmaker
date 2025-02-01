import torch
from torchlensmaker.physics import (
    refraction,
    reflection,
)
from torch.nn.functional import normalize

import itertools
import pytest


@pytest.fixture
def random_rays_2D():
    N = 10000
    rays = normalize(torch.rand(N, 2) * 2 - 1)
    raw_normals = normalize(torch.rand(N, 2) * 2 - 1)

    # Keep normals pointing against the rays
    dot = torch.sum(raw_normals * rays, dim=1)
    normals = torch.where(
        (dot > 0).unsqueeze(1).expand_as(raw_normals), -raw_normals, raw_normals
    )

    assert torch.all(torch.sum(normals * rays, dim=1) <= 0)

    return rays, normals


def nspace(N):
    n1space = [1.0, torch.full((N,), 1.0), 1.5, torch.full((N,), 1.5)]
    n2space = n1space

    return itertools.product(n1space, n2space)


def test_refraction2D_clamp(random_rays_2D) -> None:
    rays, normals = random_rays_2D
    N, D = rays.shape

    for n1, n2 in nspace(N):
        refracted, _ = refraction(rays, normals, n1, n2, "clamp")
        assert refracted.shape == rays.shape
        assert torch.allclose(
            torch.linalg.vector_norm(refracted, dim=1), torch.tensor(1.0)
        )
        dot = torch.sum(rays * refracted, dim=1)
        assert torch.all(dot >= 0)


def test_refraction2D_nan(random_rays_2D) -> None:
    rays, normals = random_rays_2D
    N, D = rays.shape

    for n1, n2 in nspace(N):
        refracted, valid = refraction(rays, normals, n1, n2, "nan")
        assert refracted.shape == rays.shape
        isfinite = torch.isfinite(refracted).all(dim=1)
        assert torch.all(isfinite == valid)
        assert torch.allclose(
            torch.linalg.vector_norm(refracted[valid], dim=1), torch.tensor(1.0)
        )
        dot = torch.sum(rays[valid] * refracted[valid], dim=1)
        assert torch.all(dot >= 0)


def test_refraction_reflect_2D(random_rays_2D) -> None:
    rays, normals = random_rays_2D
    N, D = rays.shape

    for n1, n2 in nspace(N):
        refracted, valid = refraction(rays, normals, n1, n2, "reflect")
        assert refracted.shape == rays.shape
        assert torch.allclose(
            torch.linalg.vector_norm(refracted, dim=1), torch.tensor(1.0)
        )
        dot = torch.sum(rays[valid] * refracted[valid], dim=1)
        assert torch.all(dot >= 0)


def test_refraction_drop_2D(random_rays_2D) -> None:
    rays, normals = random_rays_2D
    N, D = rays.shape

    for n1, n2 in nspace(N):
        refracted, valid = refraction(rays, normals, n1, n2, "drop")
        assert refracted.shape[0] <= N
        assert refracted.shape[1] == D
        assert torch.allclose(
            torch.linalg.vector_norm(refracted, dim=1), torch.tensor(1.0)
        )
        dot = torch.sum(rays[valid] * refracted, dim=1)
        assert torch.all(dot >= 0)
