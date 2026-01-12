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

from torchlensmaker.surfaces.local_surface import LocalSurface
from torchlensmaker.new_kinematics.homogeneous_geometry import HomMatrix, transform_points, transform_vectors

Tensor = torch.Tensor


def intersect(
    surface: LocalSurface,
    P: Tensor,
    V: Tensor,
    hom: HomMatrix,
    hom_inv: HomMatrix
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Surface-rays collision detection

    Find collision points and normal vectors for the intersection of rays P+tV with
    a surface and a transform applied to that surface.

    Args:
        P: (N, 2|3) tensor, rays origins
        V: (N, 2|3) tensor, rays vectors
        surface: surface to collide with
        hom: direct transform homogeneous matrix
        hom_inv: inverse transform homogeneous matrix

    Returns:
        t: valid collision distances
        normals: valid surface normals at the collision points
        valid: bool tensor (N,) indicating which rays do collide with the surface
    """

    assert P.shape[0] == V.shape[0]
    assert P.shape[1] == V.shape[1]
    assert P.shape[1] in {2, 3}

    # Special case for zero rays
    if P.shape[0] == 0:
        return torch.zeros(P.shape[0], dtype=P.dtype), torch.zeros_like(V), torch.full((P.shape[0],), False)

    # Convert rays to surface local frame
    Ps = transform_points(hom_inv, P)
    Vs = transform_vectors(hom_inv, V)

    # Collision detection in the surface local frame
    t, local_normals, valid = surface.local_collide(Ps, Vs)

    # Compute collision points and convert normals to global frame
    normals = transform_vectors(hom, local_normals)

    # A surface always has two opposite normals, so keep the one pointing
    # against the ray, because that's what we need for refraction / reflection
    # i.e. the normal such that dot(normal, ray) < 0
    dot = torch.sum(normals * V, dim=1)
    opposite_normals = torch.where(
        (dot > 0).unsqueeze(1).expand_as(normals), -normals, normals
    )

    assert valid.shape == (P.shape[0],)
    assert torch.all(torch.isfinite(opposite_normals))
    assert torch.all(torch.isfinite(valid))

    return t, opposite_normals, valid
