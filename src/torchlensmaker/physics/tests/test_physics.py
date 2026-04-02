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

import itertools

import pytest
import torch
from torch.nn.functional import normalize

from torchlensmaker.physics.physics import reflection, refraction
from torchlensmaker.types import Batch2DTensor


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


def test_refraction_3D_clamp(
    random_rays_3D: tuple[Batch2DTensor, Batch2DTensor],
) -> None:
    rays, normals = random_rays_3D
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


def test_refraction_3D_nan(
    random_rays_3D: tuple[Batch2DTensor, Batch2DTensor],
) -> None:
    rays, normals = random_rays_3D
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


def test_refraction_3D_reflect(
    random_rays_3D: tuple[Batch2DTensor, Batch2DTensor],
) -> None:
    rays, normals = random_rays_3D
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


def test_refraction_3D_drop(
    random_rays_3D: tuple[Batch2DTensor, Batch2DTensor],
) -> None:
    rays, normals = random_rays_3D
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


def test_refraction_coplanarity_3D(
    random_rays_3D: tuple[Batch2DTensor, Batch2DTensor],
) -> None:
    """
    In 3D, the incident ray, surface normal, and refracted ray must be coplanar.
    Verified by checking that the refracted ray has no component perpendicular
    to the plane spanned by the incident ray and normal.
    """
    rays, normals = random_rays_3D
    N = rays.shape[0]

    # Use clamp and reflect modes since they always return N outputs
    for mode in ("clamp", "reflect"):
        for n1, n2 in nspace(N):
            refracted, _ = refraction(rays, normals, n1, n2, mode)

            # The plane normal is perpendicular to both the incident ray and surface normal
            plane_normal = normalize(torch.linalg.cross(rays, normals), dim=1)

            # Refracted ray must lie in the same plane: no component along plane_normal
            out_of_plane = torch.sum(refracted * plane_normal, dim=1)
            assert torch.allclose(out_of_plane, torch.zeros(N), atol=1e-5), (
                f"mode={mode}, n1={n1}, n2={n2}: refracted ray is not coplanar with incident ray and normal"
            )


def test_refraction_reversibility_2D(
    random_rays_2D: tuple[Batch2DTensor, Batch2DTensor],
) -> None:
    """
    Refracting from n1->n2 then back n2->n1 through the same surface
    should recover the original ray direction.

    Only checked for rays where neither the forward nor the reverse refraction
    hits the critical angle (TIR). The 'nan' mode propagates NaN for TIR rays,
    which naturally marks them as invalid in the second step.
    """
    rays, normals = random_rays_2D
    N = rays.shape[0]

    n_pairs = [
        (torch.tensor(1.0), torch.tensor(1.5)),
        (torch.tensor(1.5), torch.tensor(1.0)),
        (torch.tensor(1.3), torch.tensor(1.7)),
        (torch.full((N,), 1.0), torch.full((N,), 1.5)),
    ]

    for n1, n2 in n_pairs:
        refracted, valid_fwd = refraction(rays, normals, n1, n2, "nan")
        re_refracted, valid_rev = refraction(refracted, normals, n2, n1, "nan")
        both_valid = valid_fwd & valid_rev
        # atol=1e-3 accounts for float32 precision loss near the critical angle
        assert torch.allclose(re_refracted[both_valid], rays[both_valid], atol=1e-3)


def test_refraction_reversibility_3D(
    random_rays_3D: tuple[Batch2DTensor, Batch2DTensor],
) -> None:
    """
    Refracting from n1->n2 then back n2->n1 through the same surface
    should recover the original ray direction.

    Only checked for rays where neither the forward nor the reverse refraction
    hits the critical angle (TIR). The 'nan' mode propagates NaN for TIR rays,
    which naturally marks them as invalid in the second step.
    """
    rays, normals = random_rays_3D
    N = rays.shape[0]

    n_pairs = [
        (torch.tensor(1.0), torch.tensor(1.5)),
        (torch.tensor(1.5), torch.tensor(1.0)),
        (torch.tensor(1.3), torch.tensor(1.7)),
        (torch.full((N,), 1.0), torch.full((N,), 1.5)),
    ]

    for n1, n2 in n_pairs:
        refracted, valid_fwd = refraction(rays, normals, n1, n2, "nan")
        re_refracted, valid_rev = refraction(refracted, normals, n2, n1, "nan")
        both_valid = valid_fwd & valid_rev
        # atol=1e-3 accounts for float32 precision loss near the critical angle
        assert torch.allclose(re_refracted[both_valid], rays[both_valid], atol=1e-3)
