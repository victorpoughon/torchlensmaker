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

from typing import Any, NamedTuple

from torchlensmaker.core.base_module import BaseModule
from torchlensmaker.types import (
    BatchNDTensor,
    BatchTensor,
    MaskTensor,
    ScalarTensor,
    Tf,
)


class SurfaceElementOutput(NamedTuple):
    """
    Return type of all surface elements

    Evaluating a surface module realizes three things:
        * ray-surface collisions
        * the kinematic transform with the joint added by the surface
        * the surface transform
    """

    t: BatchTensor  # P+tV are the collision points
    normals: BatchNDTensor  # unit normals at the collision points (global frame)
    valid: MaskTensor  # boolean mask for valid collisions
    points_local: BatchNDTensor  # collision points (surface frame)
    points_global: BatchNDTensor  # collision points (global frame)
    rsm: BatchTensor  # Ray-surface minimum
    tf_surface: Tf  # surface transform
    tf_next: Tf  # next transform of the kinematic chain


class SurfaceElement(BaseModule):
    """
    Abstract base class for surfaces
    """

    def outer_extent(self, anchor: ScalarTensor) -> ScalarTensor:
        """
        X coordinate of the surface at distance r from the optical axis. Raises
        if the surface is not axially symmetric, or if the surface can't compute
        this value.

        Args:
            anchor: distance from the optical axis in normalized coordinates
        """
        raise NotImplementedError

    def forward(self, P: BatchTensor, V: BatchTensor, tf: Tf) -> SurfaceElementOutput:
        raise NotImplementedError

    def render(self) -> Any:
        raise NotImplementedError
