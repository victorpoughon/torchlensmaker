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

from typing import Any, Self

import torch.nn as nn

from torchlensmaker.core.base_module import BaseModule
from torchlensmaker.types import (
    BatchNDTensor,
    BatchTensor,
    Direction,
    MaskTensor,
    ScalarTensor,
    Tf,
)


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

    def forward(
        self, P: BatchTensor, V: BatchTensor, tf: Tf, direction: Direction
    ) -> tuple[BatchTensor, BatchNDTensor, MaskTensor, Tf, Tf]:
        raise NotImplementedError

    def render(self) -> Any:
        raise NotImplementedError
