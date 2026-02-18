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

import torch
from torch.nn.functional import normalize

from torchlensmaker.physics.physics import refraction, reflection
from torchlensmaker.types import Batch2DTensor

import itertools
import pytest


# TODO test refraction in 3D


@pytest.fixture
def random_rays_2D() -> tuple[Batch2DTensor, Batch2DTensor]:
    N = 10000
    rays = normalize(torch.rand(N, 2) * 2 - 1)
    raw_normals = normalize(torch.rand(N, 2) * 2 - 1)

    # Keep normals pointing against the rays
    dot = torch.sum(raw_normals * rays, dim=-1)
    normals = torch.where(
        (dot > 0).unsqueeze(-1).expand_as(raw_normals), -raw_normals, raw_normals
    )

    assert torch.all(torch.sum(normals * rays, dim=-1) <= 0)
    return rays, normals


@pytest.fixture
def random_rays_3D() -> tuple[Batch2DTensor, Batch2DTensor]:
    N = 10000
    rays = normalize(torch.rand(N, 3) * 2 - 1)
    raw_normals = normalize(torch.rand(N, 3) * 2 - 1)

    # Keep normals pointing against the rays
    dot = torch.sum(raw_normals * rays, dim=-1)
    normals = torch.where(
        (dot > 0).unsqueeze(-1).expand_as(raw_normals), -raw_normals, raw_normals
    )

    assert torch.all(torch.sum(normals * rays, dim=-1) <= 0)
    return rays, normals


def test_reflection_2D(
    random_rays_2D: tuple[Batch2DTensor, Batch2DTensor],
) -> None:
    rays, normals = random_rays_2D

    reflected = reflection(rays, normals)

    assert reflected.shape == rays.shape
    assert reflected.dtype == rays.dtype
    assert reflected.device == rays.device
    assert torch.all(torch.sum(normals * reflected, dim=-1) >= 0)


def test_reflection_3D(
    random_rays_3D: tuple[Batch2DTensor, Batch2DTensor],
) -> None:
    rays, normals = random_rays_3D

    reflected = reflection(rays, normals)

    assert reflected.shape == rays.shape
    assert reflected.dtype == rays.dtype
    assert reflected.device == rays.device
    assert torch.all(torch.sum(normals * reflected, dim=-1) >= 0)


def nspace(N: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    n1space = [
        torch.tensor(1.0),
        torch.full((N,), 1.0),
        torch.tensor(1.5),
        torch.full((N,), 1.5),
    ]
    n2space = n1space

    return list(itertools.product(n1space, n2space))


def test_refraction_2D_clamp(
    random_rays_2D: tuple[Batch2DTensor, Batch2DTensor],
) -> None:
    rays, normals = random_rays_2D
    N, D = rays.shape

    for n1, n2 in nspace(N):
        refracted, _ = refraction(rays, normals, n1, n2, "clamp")
        assert refracted.shape == rays.shape
        assert refracted.dtype == rays.dtype
        assert refracted.device == rays.device
        assert torch.allclose(
            torch.linalg.vector_norm(refracted, dim=1), torch.tensor(1.0)
        )
        dot = torch.sum(rays * refracted, dim=1)
        assert torch.all(dot >= 0)


def test_refraction_2D_nan(random_rays_2D: tuple[Batch2DTensor, Batch2DTensor]) -> None:
    rays, normals = random_rays_2D
    N, D = rays.shape

    for n1, n2 in nspace(N):
        refracted, valid = refraction(rays, normals, n1, n2, "nan")
        assert refracted.shape == rays.shape
        assert refracted.dtype == rays.dtype
        assert refracted.device == rays.device
        isfinite = torch.isfinite(refracted).all(dim=1)
        assert torch.all(isfinite == valid)
        assert torch.allclose(
            torch.linalg.vector_norm(refracted[valid], dim=1), torch.tensor(1.0)
        )
        dot = torch.sum(rays[valid] * refracted[valid], dim=1)
        assert torch.all(dot >= 0)


def test_refraction_2D_reflect(
    random_rays_2D: tuple[Batch2DTensor, Batch2DTensor],
) -> None:
    rays, normals = random_rays_2D
    N, D = rays.shape

    for n1, n2 in nspace(N):
        refracted, valid = refraction(rays, normals, n1, n2, "reflect")
        assert refracted.shape == rays.shape
        assert refracted.dtype == rays.dtype
        assert refracted.device == rays.device
        assert torch.allclose(
            torch.linalg.vector_norm(refracted, dim=1), torch.tensor(1.0)
        )
        dot = torch.sum(rays[valid] * refracted[valid], dim=1)
        assert torch.all(dot >= 0)


def test_refraction_2D_drop(
    random_rays_2D: tuple[Batch2DTensor, Batch2DTensor],
) -> None:
    rays, normals = random_rays_2D
    N, D = rays.shape

    for n1, n2 in nspace(N):
        refracted, valid = refraction(rays, normals, n1, n2, "drop")
        assert refracted.shape[0] <= N
        assert refracted.shape[1] == D
        assert refracted.dtype == rays.dtype
        assert refracted.device == rays.device
        assert torch.allclose(
            torch.linalg.vector_norm(refracted, dim=1), torch.tensor(1.0)
        )
        dot = torch.sum(rays[valid] * refracted, dim=1)
        assert torch.all(dot >= 0)
