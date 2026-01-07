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
import math

from torchlensmaker.core.geometry import within_radius


class Outline:
    "An outline limits the extent of a 3D surface in the local YZ plane"

    def contains(self, points: torch.Tensor, tol: float = 1e-6) -> torch.Tensor:
        raise NotImplementedError

    def max_radius(self) -> float:
        "Furthest distance to the X axis that's within the outline"
        raise NotImplementedError

    def clip_planes(self) -> list[list[float]]:
        "Express the outline as 3D clip planes"
        raise NotImplementedError


class SquareOutline(Outline):
    "Square outline around the X axis"

    def __init__(self, side_length: float):
        self.side_length = side_length

    def contains(self, points: torch.Tensor, tol: float = 1e-6) -> torch.Tensor:
        dim = points.shape[1]
        if dim == 2:
            return torch.le(torch.abs(points[:, 1]), self.max_radius())
        else:
            return (
                torch.maximum(torch.abs(points[:, 1]), torch.abs(points[:, 2]))
                < self.side_length / 2
            )

    def max_radius(self) -> float:
        return math.sqrt(2) * self.side_length / 2

    def clip_planes(self) -> list[list[float]]:
        a = self.side_length / 2
        return [
            [0.0, -1.0, 0.0, a],
            [0.0, 1.0, 0.0, a],
            [0.0, 0.0, -1.0, a],
            [0.0, 0.0, 1.0, a],
        ]


class CircularOutline(Outline):
    "Fixed distance to the X axis"

    def __init__(self, diameter: float):
        self.diameter = diameter

    def contains(self, points: torch.Tensor, tol: float = 1e-6) -> torch.Tensor:
        # TODO improve this with rtol / atol
        return within_radius(self.diameter / 2 + tol, points)

    def max_radius(self) -> float:
        return self.diameter / 2

    def clip_planes(self) -> list[list[float]]:
        return []
