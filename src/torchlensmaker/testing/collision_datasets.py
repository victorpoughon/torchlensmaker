# This file is part of Torch Lens Maker
# Copyright (C) 2025 Victor Poughon
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
import torch.nn as nn

from torchlensmaker.core.geometry import unit_vector, rotated_unit_vector
from torchlensmaker.core.surfaces import ImplicitSurface, LocalSurface
from torchlensmaker.core.rot2d import perpendicular2d, rot2d
import math

from typing import Callable, TypeAlias
from dataclasses import dataclass, replace

Tensor: TypeAlias = torch.Tensor


def make_samples(
    surface: LocalSurface, dim: int, N: int, epsilon: float = 0.0
) -> tuple[Tensor, Tensor]:
    "Generate samples and normals for a surface"

    if dim == 2:
        samples = surface.samples2D_full(N, epsilon=epsilon)
    else:
        root = math.ceil(math.sqrt(N))
        samples = make_samples3D(surface.samples2D_full(root, epsilon=epsilon), root)
    return samples, surface.normals(samples)


CollisionDataset: TypeAlias = tuple[Tensor, Tensor]

class RayGenerator:
    def __call__(self, surface: LocalSurface) -> CollisionDataset:
        raise NotImplementedError


@dataclass
class NormalRays(RayGenerator):
    "Ray generator for rays normal to the surface"
    dim: int
    N: int
    offset: float
    epsilon: float

    def __call__(self, surface: LocalSurface) -> CollisionDataset:
        samples, normals = make_samples(surface, self.dim, self.N, self.epsilon)

        # offset along V
        P = samples + self.offset * normals
        V = normals

        return P, V


@dataclass
class TangentRays(RayGenerator):
    "Ray generator for rays tangent to the surface"
    dim: int
    N: int
    distance: float
    epsilon: float

    def __call__(self, surface: LocalSurface) -> CollisionDataset:
        assert self.dim == 2
        samples, normals = make_samples(surface, self.dim, self.N, self.epsilon)

        # Points on the surface, offset by 'distance' along the gradient
        P = samples + self.distance * normals

        # Vectors are perpendicular to the gradient
        V = nn.functional.normalize(perpendicular2d(normals))

        return P, V


@dataclass
class RandomRays:
    "Ray generator for random direction rays"
    dim: int
    N: int
    offset: float
    epsilon: float
    
    def __call__(self, surface: LocalSurface) -> CollisionDataset:
        samples, _ = make_samples(surface, self.dim, self.N, self.epsilon)

        assert self.dim == 2 # TODO support 3D

        theta = 2 * math.pi * torch.rand((self.N,))
        V = rot2d(torch.tensor([1.0, 0.0]), theta).to(dtype=surface.dtype)

        P = samples + self.offset * V

        return P, V


@dataclass
class FixedRays(RayGenerator):
    "Ray generator for rays with a fixed direction"
    dim: int
    N: int
    direction: Tensor
    offset: float
    epsilon: float

    def __call__(self, surface: LocalSurface) -> CollisionDataset:
        # normalize direction
        assert self.direction.numel() == self.dim
        direction = torch.nn.functional.normalize(self.direction, dim=0)

        samples, _ = make_samples(surface, self.dim, self.N, self.epsilon)

        assert torch.all(torch.isfinite(samples)), (surface.__dict__, self.dim, self.N, self.epsilon)

        V = torch.tile(direction, (samples.shape[0], 1)).to(dtype=surface.dtype)
        P = (samples + self.offset * V).to(dtype=surface.dtype)

        return P, V


@dataclass
class OrbitalRays(RayGenerator):
    "Ray generator for rays tangent to a circle centered at the origin"

    dim: int
    N: int
    radius: float
    offset: float
    epsilon: float

    def __call__(self, surface: LocalSurface) -> CollisionDataset:
        if self.dim == 2:
            theta = torch.linspace(0., 2*torch.pi, self.N, dtype=surface.dtype)

            points = torch.stack((
                self.radius * surface.bounding_radius() * torch.cos(theta),
                self.radius * surface.bounding_radius() * torch.sin(theta)
            ), dim=-1)

            V = rotated_unit_vector(torch.tensor(torch.pi / 2, dtype=surface.dtype) + theta, self.dim)

            return points, V
        else:
            raise NotImplementedError # TODO


def make_offset_rays(P: torch.Tensor, V: torch.Tensor, tspace: torch.Tensor) -> tuple[Tensor, Tensor]:
    "Duplicate rays by moving the origin P along V by t units"

    assert tspace.dim() == 1

    newP = torch.cat([P + t*V for t in tspace], dim=0)
    newV = torch.tile(V, dims=(tspace.shape[0], 1))

    return newP, newV


def make_samples3D(samples2D: torch.Tensor, M: int) -> torch.Tensor:
    """
    Given a tensor of points of shape (N, 2), make a tensor of shape (M*N, 3) by
    axial rotation around the X axis and at M angles linearly spaced around the
    circle
    """
    step = 2 * torch.pi / M
    angles = torch.linspace(0, (M - 1) * step, M, device=samples2D.device)
    cosθ = torch.cos(angles)
    sinθ = torch.sin(angles)

    # Split coordinates and expand for broadcasting
    x = samples2D[:, 0].unsqueeze(1).repeat(1, M)  # (N, M)
    y = samples2D[:, 1].unsqueeze(1).repeat(1, M)  # (N, M)

    # Calculate rotated coordinates
    y_rotated = y * cosθ.unsqueeze(0)  # (N, M)
    z_rotated = y * sinθ.unsqueeze(0)  # (N, M)

    # Combine and reshape
    rotated_points = torch.stack([x, y_rotated, z_rotated], dim=2).view(-1, 3)

    return rotated_points
