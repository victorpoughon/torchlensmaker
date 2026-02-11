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


from jaxtyping import Float
from typing import Self

from torchlensmaker.types import (
    ScalarTensor,
    BatchTensor,
    BatchNDTensor,
    MaskTensor,
    HomMatrix,
)

from torchlensmaker.core.tensor_manip import init_param

from .surface_kernels import SphereC2DSurfaceKernel


class SphereC(nn.Module):
    """
    Spherical surface (2D or 3D) parameterized by lens diameter and curvature.

    Represented by a sag function and raytraced by implicit solver
    Support for anchors and scale.
    """

    def __init__(
        self,
        diameter: float | ScalarTensor,
        C: float | ScalarTensor | nn.Parameter,
        *,
        anchors: tuple[float, float] | Float[torch.Tensor, " 2"] = (0.0, 0.0),
        scale: float | ScalarTensor = 1.0,
        trainable: bool = True,
    ):
        super().__init__()
        self.diameter = init_param(self, "diameter", diameter, False)
        self.C = init_param(self, "C", C, trainable)
        self.anchors = init_param(self, "anchors", anchors, False)
        self.scale = init_param(self, "scale", scale, False)
        self.func2d = SphereC2DSurfaceKernel()

    def forward(
        self, P: BatchTensor, V: BatchTensor, dfk: BatchTensor, ifk: BatchTensor
    ) -> tuple[BatchTensor, BatchNDTensor, MaskTensor, HomMatrix, HomMatrix, HomMatrix, HomMatrix]:
        return self.func2d.forward(
            P, V, dfk, ifk, self.diameter, self.C, self.anchors, self.scale
        )
