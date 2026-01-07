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
import torch.nn as nn
from typing import Optional, Any


class LocalSurface:
    """
    Abstract base class for all surfaces
    """

    def __init__(self, dtype: torch.dtype = torch.float64):
        self.dtype = dtype

    def parameters(self) -> dict[str, nn.Parameter]:
        raise NotImplementedError

    def zero(self, dim: int) -> torch.Tensor:
        "N-dimensional zero point"
        return torch.zeros((dim,), dtype=self.dtype)

    def extent(self, dim: int) -> torch.Tensor:
        "N-dimensional extent point"
        return torch.cat(
            (self.extent_x().unsqueeze(0), torch.zeros(dim - 1, dtype=self.dtype)),
            dim=0,
        )

    def extent_x(self) -> torch.Tensor:
        """
        Extent along the X axis
        i.e. X coordinate of the point on the surface that is furthest along the X axis
        """
        raise NotImplementedError

    def normals(self, points: torch.Tensor) -> torch.Tensor:
        """
        Unit vectors normal to the surface at input points of shape (..., D)
        All dimensions except the last one are batch dimensions
        Input points are expected to be on, or at least near, the surface
        """
        raise NotImplementedError

    def contains(
        self, points: torch.Tensor, tol: Optional[float] = None
    ) -> torch.Tensor:
        """
        Test if points belong to the surface

        Args:
            * points: Input points of shape (..., D)
            * tol: optional tolerance parameter (default None). None means use an internal default based on dtype

        Returns:
            * boolean mask of shape points.shape[:-1]
        """
        raise NotImplementedError

    def samples2D_half(self, N: int, epsilon: float) -> torch.Tensor:
        "Generate 2D samples on the half positive domain"
        raise NotImplementedError

    def samples2D_full(self, N: int, epsilon: float) -> torch.Tensor:
        "Generate 2D samples on the full domain"
        raise NotImplementedError

    def bounding_radius(self) -> float:
        """
        Any point on the surface has a distance to the center that is less
        than (or equal) to the bounding radius
        """
        raise NotImplementedError

    def local_collide(
        self, P: torch.Tensor, V: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Find collision points and surface normals of ray-surface intersection
        for parametric rays P+tV expressed in the surface local frame.

        Returns:
            t: Value of parameter t such that P + tV is on the surface
            normals: Normal unit vectors to the surface at the collision points
            valid: Bool tensor indicating which rays do collide with the surface
        """
        raise NotImplementedError

    def to_dict(self, dim: int) -> dict[str, Any]:
        """
        Convert to a dictionary for JSON serialization
        """
        raise NotImplementedError
