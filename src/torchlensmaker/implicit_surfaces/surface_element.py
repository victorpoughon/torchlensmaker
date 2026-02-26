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

from typing import Any
import torch.nn as nn
from typing import Self
from torchlensmaker.types import Tf, BatchTensor, BatchNDTensor, MaskTensor, ScalarTensor


class SurfaceElement(nn.Module):
    """
    Abstract base class for surfaces
    """

    def clone(self, **overrides) -> Self:
        raise NotImplementedError
    
    def outer_extent(self, r: ScalarTensor) -> ScalarTensor | None:
        """
        X coordinate of the surface at distance r from the optical axis. None if
        the surface is not axially symmetric, or if the surface can't compute
        this value.
        """
        return None

    def forward(
        self, P: BatchTensor, V: BatchTensor, tf: Tf
    ) -> tuple[BatchTensor, BatchNDTensor, MaskTensor, Tf, Tf]:
        raise NotImplementedError

    def render(self) -> Any:
        raise NotImplementedError
