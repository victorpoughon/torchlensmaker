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
from enum import Enum
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


class Direction(Enum):
    """
    Direction of propagation

    The two possible directions are effectively forwards / backwards, but we use
    the vocabulary prograde / retrograde instead, to avoid any confusion with
    other terminology which already uses common terms:
    
      * forward / backwards (pytorch autograd)
      * direct / inverse (geometric transforms)
      * forward / inverse (kinematics)
    
    and we use an enum to avoid the "boolean argument" footgun.
    """

    PROGRADE = "prograde"
    RETROGRADE = "retrograde"

    def is_prograde(self) -> bool:
        return self is self.PROGRADE

    def is_retrograde(self) -> bool:
        return self is self.RETROGRADE

    def as_scale(self) -> float:
        return 1.0 if self.is_prograde() else -1.0


@dataclass
class Tf:
    """
    Geometric transform (2D or 3D)
     
    Represented by a pair of homogeneous coordinate matrices for the direct and
    inverse transforms.
    """
    direct: HomMatrix
    inverse: HomMatrix

    def pdim(self) -> int:
        "Physical dimensions (2 or 3)"
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

    def flip(self) -> Self:
        return Tf(self.inverse, self.direct)

    def flipif(self, flip: bool) -> Self:
        return self.flip() if flip else self
