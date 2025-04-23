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

from torchlensmaker.optical_data import OpticalData
from torchlensmaker.core.tensor_manip import to_tensor, filter_optional_tensor
from torchlensmaker.core.transforms import (
    TransformBase,
    TranslateTransform,
    LinearTransform,
    forward_kinematic,
)
from torchlensmaker.core.rot2d import rotation_matrix_2D
from torchlensmaker.core.rot3d import euler_angles_to_matrix

from typing import TypeAlias

Tensor: TypeAlias = torch.Tensor

"""
SubChain
AbsoluteTransform
RelativeTransform

Gap

Rotate3D
Translate2D
Translate3D
"""


class SubChain(nn.Module):
    def __init__(self, *children: nn.Module):
        super().__init__()
        self._sequential = nn.Sequential(*children)

    def forward(self, inputs: OpticalData) -> OpticalData:
        output = self._sequential(inputs)
        return output.replace(transforms=inputs.transforms)


class AbsoluteTransform(nn.Module):
    def __init__(self, tf: TransformBase):
        super().__init__()
        self.tf = tf

    def forward(self, inputs: OpticalData) -> OpticalData:
        return inputs.replace(transforms=[self.tf])


class RelativeTransform(nn.Module):
    def tf(self, dim: int, dtype: torch.dtype) -> TransformBase:
        "Transform that gets appended to the kinematic chain by this element"
        raise NotImplementedError

    def forward(self, inputs: OpticalData) -> OpticalData:
        dim, dtype = inputs.dim, inputs.dtype
        return inputs.replace(
            transforms=inputs.transforms + [self.tf(dim, dtype)],
        )


class Gap(RelativeTransform):
    def __init__(self, offset: float | int | Tensor):
        super().__init__()
        assert isinstance(offset, (float, int, torch.Tensor))
        if isinstance(offset, torch.Tensor):
            assert offset.dim() == 0

        # Gap is always stored as float64, but it's converted to the sampling
        # dtype when creating the corresponding transform in forward()
        self.offset = torch.as_tensor(offset, dtype=torch.float64)

    def tf(self, dim: int, dtype: torch.dtype) -> TransformBase:
        translate_vector = torch.cat(
            (
                self.offset.unsqueeze(0).to(dtype=dtype),
                torch.zeros(dim - 1, dtype=dtype),
            )
        )
        return TranslateTransform(translate_vector)


class Rotate3D(RelativeTransform):
    "3D rotation (in degrees)"

    def __init__(self, angles: tuple[float | int, float | int] | Tensor):
        super().__init__()

        if not isinstance(angles, torch.Tensor):
            angles = torch.as_tensor(angles, dtype=torch.float64)

        self.angles = angles

    def tf(self, dim: int, dtype: torch.dtype) -> TransformBase:
        assert dim == 3
        radangles = torch.deg2rad(self.angles)
        M = euler_angles_to_matrix(
            torch.stack((torch.tensor(0, dtype=dtype), radangles[0], radangles[1])),
            "XZY",
        ).to(dtype=dtype)  # TODO need to support dtype in euler_angles_to_matrix
        return LinearTransform(M, M.T)


class Rotate2D(RelativeTransform):
    "2D rotation (in degrees)"

    def __init__(self, angle: float | int | Tensor):
        super().__init__()
        self._angle = to_tensor(angle)

    def tf(self, dim: int, dtype: torch.dtype) -> OpticalData:
        assert dim == 2
        M = rotation_matrix_2D(torch.deg2rad(self._angle))
        return LinearTransform(M, M.T)


class Translate3D(RelativeTransform):
    def __init__(
        self,
        x: Tensor | float = 0.0,
        y: Tensor | float = 0.0,
        z: Tensor | float = 0.0,
    ):
        super().__init__()
        self._x, self._y, self._z = to_tensor(x), to_tensor(y), to_tensor(z)

    def tf(self, dim: int, dtype: torch.dtype) -> TransformBase:
        return TranslateTransform(torch.stack((self._x, self._y, self._z), dim=-1))


class Translate2D(RelativeTransform):
    def __init__(
        self,
        x: Tensor | float = 0.0,
        r: Tensor | float = 0.0,
    ):
        super().__init__()
        self._x, self._r = to_tensor(x), to_tensor(r)

    def tf(self, dim: int, dtype: torch.dtype) -> TransformBase:
        return TranslateTransform(torch.stack((self._x, self._r), dim=-1))


class MixedDim(nn.Module):
    "2D or 3D branch"

    def __init__(self, dim2: nn.Module, dim3: nn.Module):
        super().__init__()
        self._dim2 = dim2
        self._dim3 = dim3

    def forward(self, inputs: OpticalData) -> OpticalData:
        if inputs.dim == 2:
            return self._dim2(inputs)
        else:
            return self._dim3(inputs)
