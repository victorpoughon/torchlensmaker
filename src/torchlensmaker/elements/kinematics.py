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

from torchlensmaker.elements.sequential import SequentialElement
from torchlensmaker.optical_data import OpticalData
from torchlensmaker.core.tensor_manip import to_tensor
from torchlensmaker.core.transforms import (
    TransformBase,
    TranslateTransform,
    LinearTransform,
)
from torchlensmaker.core.rot2d import rotation_matrix_2D
from torchlensmaker.core.rot3d import euler_angles_to_matrix
from typing import TypeAlias, Sequence, cast

Tensor: TypeAlias = torch.Tensor
KinematicChain: TypeAlias = Sequence[TransformBase]


class KinematicElement(SequentialElement):
    """Base class for kinematic elements
    """

    def forward(self, chain: KinematicChain) -> KinematicChain:
        raise NotImplementedError

    def sequential(self, data: OpticalData) -> OpticalData:
        return data.replace(transforms=self(data.transforms))


class AbsoluteTransform(KinematicElement):
    def __init__(self, tf: TransformBase):
        super().__init__()
        self.tf = tf

    def forward(self, chain: KinematicChain) -> KinematicChain:
        return [self.tf]


class AbsolutePosition(KinematicElement):
    def __init__(
        self,
        x: int | float | Tensor = 0.0,
        y: int | float | Tensor = 0.0,
        z: int | float | Tensor = 0.0,
    ):
        super().__init__()
        self.x, self.y, self.z = to_tensor(x), to_tensor(y), to_tensor(z)

    def forward(self, chain: KinematicChain) -> KinematicChain:
        return [TranslateTransform(torch.stack((self.x, self.y, self.z), dim=-1))]


class RelativeTransform(KinematicElement):
    def tf(self, dim: int, dtype: torch.dtype) -> TransformBase:
        "Transform that gets appended to the kinematic chain by this element"
        raise NotImplementedError

    def forward(self, chain: KinematicChain) -> KinematicChain:
        assert len(chain) > 0
        # There is some mypy BS here about Sequence and covariant typing
        return chain + [self.tf(chain[0].dim, chain[0].dtype)]  # type: ignore[operator,no-any-return]


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

    def __init__(self, y: int | float | Tensor = 0.0, z: int | float | Tensor = 0.0):
        super().__init__()
        self.y = to_tensor(y)
        self.z = to_tensor(z)

    def tf(self, dim: int, dtype: torch.dtype) -> TransformBase:
        assert dim == 3

        radangles = torch.deg2rad(torch.stack((self.y, self.z)))
        M = euler_angles_to_matrix(
            torch.stack((torch.tensor(0, dtype=dtype), radangles[0], radangles[1])),
            "XYZ",
        ).to(dtype=dtype)  # TODO need to support dtype in euler_angles_to_matrix
        return LinearTransform(M, M.T)


class Rotate2D(RelativeTransform):
    "2D rotation (in degrees)"

    def __init__(self, angle: float | int | Tensor):
        super().__init__()
        self._angle = to_tensor(angle)

    def tf(self, dim: int, dtype: torch.dtype) -> TransformBase:
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


class MixedDim(KinematicElement):
    "2D or 3D branch for a kinematic element"

    def __init__(self, dim2: KinematicChain, dim3: KinematicChain):
        super().__init__()
        self._dim2 = dim2
        self._dim3 = dim3

    def forward(self, chain: KinematicChain) -> KinematicChain:
        if chain[0].dim == 2:
            return cast(KinematicChain, self._dim2(chain))
        else:
            return cast(KinematicChain, self._dim3(chain))


class Rotate(MixedDim):
    """Mixed dimension rotation

    The second angle is ignored in 2D.
    """

    def __init__(self, angles: tuple[float | int, float | int] | Tensor):
        super().__init__(dim2=Rotate2D(angles[0]), dim3=Rotate3D(angles[0], angles[1]))


class Translate(MixedDim):
    """Mixed dimension translation

    In 2D, the z coordinate is ignored, and the y coordinate stands for the r (radial) coordinate.
    """

    def __init__(
        self,
        x: Tensor | float = 0.0,
        y: Tensor | float = 0.0,
        z: Tensor | float = 0.0,
    ):
        super().__init__(dim2=Translate2D(x, y), dim3=Translate3D(x, y, z))
