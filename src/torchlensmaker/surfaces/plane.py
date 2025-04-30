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

from torchlensmaker.surfaces.local_surface import LocalSurface
from torchlensmaker.core.geometry import unit_vector
from torchlensmaker.core.collision_detection import init_closest_origin

from torchlensmaker.core.outline import (
    Outline,
    SquareOutline,
    CircularOutline,
)

from typing import TypeAlias, Optional, Any

Tensor: TypeAlias = torch.Tensor


class Plane(LocalSurface):
    "X=0 plane"

    def __init__(self, outline: Outline, dtype: torch.dtype):
        super().__init__(dtype)
        self.outline = outline

    def parameters(self) -> dict[str, nn.Parameter]:
        return {}

    def samples2D_half(self, N: int, epsilon: float) -> Tensor:
        maxr = (1.0 - epsilon) * self.outline.max_radius()
        r = torch.linspace(0, maxr, N, dtype=self.dtype)
        return torch.stack((torch.zeros(N), r), dim=-1)

    def samples2D_full(self, N: int, epsilon: float) -> Tensor:
        maxr = (1.0 - epsilon) * self.outline.max_radius()
        r = torch.linspace(-maxr, maxr, N, dtype=self.dtype)
        return torch.stack((torch.zeros(N), r), dim=-1)

    def local_collide(self, P: Tensor, V: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        # Nominal case: rays are not perpendicular to the X axis
        mask_nominal = V[:, 0] != torch.zeros(1, dtype=V.dtype)

        t = torch.where(
            mask_nominal,
            -P[:, 0] / V[:, 0],  # Nominal case, rays aren't vertical
            init_closest_origin(self, P, V),  # Default for vertical rays
        )

        local_points = P + t.unsqueeze(1).expand_as(V) * V
        valid = self.outline.contains(local_points)
        return t, self.normals(local_points), valid

    def normals(self, points: Tensor) -> Tensor:
        batch, dim = points.shape[:-1], points.shape[-1]
        normal = -unit_vector(dim=dim, dtype=self.dtype)
        return torch.tile(normal, (*batch, 1))

    def extent_x(self) -> Tensor:
        return torch.as_tensor(0.0, dtype=self.dtype)

    def bounding_radius(self) -> float:
        return self.outline.max_radius()

    def contains(self, points: Tensor, tol: Optional[float] = None) -> Tensor:
        if tol is None:
            tol = {torch.float32: 1e-4, torch.float64: 1e-6}[self.dtype]

        return torch.logical_and(
            self.outline.contains(points, tol), torch.abs(points.select(-1, 0)) < tol
        )

    def to_dict(self, dim: int) -> dict[str, Any]:
        return {
            "type": "surface-plane",
            "radius": self.outline.max_radius(),
            "clip_planes": self.outline.clip_planes(),
        }


class SquarePlane(Plane):
    def __init__(self, side_length: float, dtype: torch.dtype = torch.float64):
        self.side_length = side_length
        super().__init__(SquareOutline(side_length), dtype)


class CircularPlane(Plane):
    "aka disk"

    def __init__(self, diameter: float, dtype: torch.dtype = torch.float64):
        self.diameter = diameter
        super().__init__(CircularOutline(diameter), dtype)
