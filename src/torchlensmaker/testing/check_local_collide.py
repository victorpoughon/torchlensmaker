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

import torchlensmaker as tlm


def check_local_collide(
    surface: tlm.LocalSurface, P: torch.Tensor, V: torch.Tensor, expected_collide: bool
) -> None:
    "Call surface.local_collide() and performs tests on the output"

    # Check that rays are the correct dtype
    assert P.dtype == surface.dtype
    assert V.dtype == surface.dtype

    # Call local_collide, rays in testing datasets are in local frame
    batch, D = P.shape[:-1], P.shape[-1]
    t, local_normals, valid = surface.local_collide(P, V)
    local_points = P + t.unsqueeze(-1).expand_as(V) * V

    # Check shapes
    assert t.dim() == len(batch) and t.shape == batch
    assert local_normals.dim() == len(batch) + 1 and local_normals.shape == (
        *batch,
        D,
    )
    assert valid.dim() == len(batch) and valid.shape == batch
    assert local_points.dim() == 2 and local_points.shape == (*batch, D)

    # Check dtypes
    assert t.dtype == surface.dtype, (P.dtype, V.dtype, t.dtype, surface.dtype)
    assert local_normals.dtype == surface.dtype
    assert valid.dtype == torch.bool
    assert local_points.dtype == surface.dtype

    # Check isfinite
    assert torch.all(torch.isfinite(t)), surface
    assert torch.all(torch.isfinite(local_normals)), local_normals
    assert torch.all(torch.isfinite(valid))
    assert torch.all(torch.isfinite(local_points))

    # Check all normals are unit vectors
    assert torch.allclose(
        torch.linalg.vector_norm(local_normals, dim=-1),
        torch.ones(1, dtype=surface.dtype),
    )

    if isinstance(surface, tlm.ImplicitSurface):
        rmse = surface.rmse(local_points)
    else:
        rmse = None

    # Check expected collision against expected_collide
    assert torch.all(surface.contains(local_points) == expected_collide), (
        str(surface),
        rmse,
    )
    assert torch.all(valid == expected_collide), surface
