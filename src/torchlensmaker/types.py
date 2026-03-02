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

from dataclasses import dataclass
from typing import TypeAlias, Self, Literal
from jaxtyping import Float, Bool, Int64
import torch


# Tensors of various types and shapes
ScalarTensor: TypeAlias = Float[torch.Tensor, ""]
BatchTensor: TypeAlias = Float[torch.Tensor, "..."]
Batch2DTensor: TypeAlias = Float[torch.Tensor, "... 2"]
Batch3DTensor: TypeAlias = Float[torch.Tensor, "... 3"]
BatchNDTensor: TypeAlias = Float[torch.Tensor, "... D"]
MaskTensor: TypeAlias = Bool[torch.Tensor, "..."]
IndexTensor: TypeAlias = Int64[torch.Tensor, "..."]
IndexNDTensor: TypeAlias = Int64[torch.Tensor, "... D"]

# Homogeneous coordinates matrix
HomMatrix: TypeAlias = Float[torch.Tensor, "D D"]

TIRMode: TypeAlias = Literal["absorb", "reflect"]

# Geometric transform (2D or 3D) represented by a pair of homogeneous coordinate
# matrices for the direct and inverse transforms
@dataclass
class Tf:
    direct: HomMatrix
    inverse: HomMatrix

    def dim(self) -> int:
        assert self.direct.shape == self.inverse.shape
        return self.direct.shape[0] - 1

    def clone(self) -> Self:
        return type(self)(self.direct.clone(), self.inverse.clone())

    @property
    def dtype(self) -> torch.dtype:
        assert self.direct.dtype == self.inverse.dtype
        return self.direct.dtype

    @property
    def shape(self) -> torch.Size:
        assert self.direct.shape == self.inverse.shape
        return self.direct.shape

    @property
    def device(self) -> torch.device:
        assert self.direct.device == self.inverse.device
        return self.direct.device
