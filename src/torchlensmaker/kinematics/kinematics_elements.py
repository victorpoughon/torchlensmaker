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

from typing import Any, Self
from jaxtyping import Float

from torchlensmaker.core.tensor_manip import to_tensor, init_param
from torchlensmaker.optical_data import OpticalData

from .homogeneous_geometry import (
    HomMatrix2D,
    HomMatrix3D,
    HomMatrix,
)
from .kinematics_kernels import (
    AbsolutePosition3DKernel,
    Rotate2DKernel,
    Rotate3DKernel,
    Translate2DKernel,
    Translate3DKernel,
    Gap2DKernel,
    Gap3DKernel,
)

# TODO this is used for lens only
from .homogeneous_geometry import kinematic_chain_append

from torchlensmaker.elements.sequential_element import SequentialElement


class KinematicElement(SequentialElement):
    def sequential(self, data: OpticalData) -> OpticalData:
        dfk, ifk = self(data.dfk, data.ifk)
        return data.replace(dfk=dfk, ifk=ifk)


class KinematicSequential(nn.Module):
    def __init__(self, *sequence: nn.Module):
        super().__init__()
        self.sequence = nn.ModuleList(sequence)

    def forward(self, dfk: HomMatrix, ifk: HomMatrix):
        for mod in self.sequence:
            dfk, ifk = mod(dfk, ifk)
        return dfk, ifk


# TODO this is used for lens only to make a "kinematic only" sequential model
# no kernel needed for now
class ExactKinematicElement2D(KinematicElement):
    def __init__(self, hom: HomMatrix2D, hom_inv: HomMatrix2D):
        super().__init__()
        self.hom = hom
        self.hom_inv = hom_inv

    def forward(
        self, dfk: HomMatrix2D, ifk: HomMatrix2D
    ) -> tuple[HomMatrix2D, HomMatrix2D]:
        return kinematic_chain_append(dfk, ifk, self.hom, self.hom_inv)

    def reverse(self) -> Self:
        return type(self)(self.hom_inv, self.hom)


class Translate2D(KinematicElement):
    def __init__(
        self,
        x: Float[torch.Tensor, ""] | float = 0.0,
        y: Float[torch.Tensor, ""] | float = 0.0,
    ):
        super().__init__()
        self.func = Translate2DKernel()
        self.x = to_tensor(x)
        self.y = to_tensor(y)

    def forward(
        self, dfk: HomMatrix2D, ifk: HomMatrix2D
    ) -> tuple[HomMatrix2D, HomMatrix2D]:
        return self.func.forward(dfk, ifk, self.x, self.y)

    def reverse(self) -> Self:
        return type(self)(-self.x, -self.y)


class TranslateVec2D(KinematicElement):
    def __init__(
        self,
        T: Float[torch.Tensor, "3"] | list[float | int],
    ):
        super().__init__()
        self.func = Translate2DKernel()
        self.T = to_tensor(T)

    def forward(
        self, dfk: HomMatrix2D, ifk: HomMatrix2D
    ) -> tuple[HomMatrix2D, HomMatrix2D]:
        return self.func.forward(dfk, ifk, *torch.unbind(self.T))


class Translate3D(KinematicElement):
    def __init__(
        self,
        x: Float[torch.Tensor, ""] | float | int = 0.0,
        y: Float[torch.Tensor, ""] | float | int = 0.0,
        z: Float[torch.Tensor, ""] | float | int = 0.0,
    ):
        super().__init__()
        self.func = Translate3DKernel()
        self.x = to_tensor(x)
        self.y = to_tensor(y)
        self.z = to_tensor(z)

    def forward(
        self, dfk: HomMatrix3D, ifk: HomMatrix3D
    ) -> tuple[HomMatrix3D, HomMatrix3D]:
        return self.func.forward(dfk, ifk, self.x, self.y, self.z)


class TranslateVec3D(KinematicElement):
    def __init__(
        self,
        T: Float[torch.Tensor, "3"] | list[float | int],
    ):
        super().__init__()
        self.func = Translate3DKernel()
        self.T = to_tensor(T)

    def forward(
        self, dfk: HomMatrix3D, ifk: HomMatrix3D
    ) -> tuple[HomMatrix3D, HomMatrix3D]:
        return self.func.forward(dfk, ifk, *torch.unbind(self.T))


class Rotate2D(KinematicElement):
    "2D rotation (in degrees)"

    def __init__(self, theta: Float[torch.Tensor, ""] | float | int = 0.0):
        super().__init__()
        self.func = Rotate2DKernel()
        self.theta = to_tensor(theta)

    def forward(
        self, dfk: HomMatrix2D, ifk: HomMatrix2D
    ) -> tuple[HomMatrix2D, HomMatrix2D]:
        return self.func.forward(dfk, ifk, self.theta)


# TODO rename to AbsolutePosition3D
class AbsolutePosition(KinematicElement):
    def __init__(
        self,
        x: Float[torch.Tensor, ""] | float | int = 0.0,
        y: Float[torch.Tensor, ""] | float | int = 0.0,
        z: Float[torch.Tensor, ""] | float | int = 0.0,
    ):
        super().__init__()
        self.func = AbsolutePosition3DKernel()
        self.x = to_tensor(x)
        self.y = to_tensor(y)
        self.z = to_tensor(z)

    def forward(
        self, dfk: HomMatrix3D, ifk: HomMatrix3D
    ) -> tuple[HomMatrix3D, HomMatrix3D]:
        return self.func.forward(dfk, ifk, self.x, self.y, self.z)


class AbsolutePosition3D(KinematicElement):
    def __init__(
        self,
        T: Float[torch.Tensor, "3"] | list[float | int],
    ):
        super().__init__()
        self.func = AbsolutePosition3DKernel()
        self.T = to_tensor(T)

    def forward(
        self, dfk: HomMatrix3D, ifk: HomMatrix3D
    ) -> tuple[HomMatrix3D, HomMatrix3D]:
        return self.func.forward(dfk, ifk, *torch.unbind(self.T))


class Rotate3D(KinematicElement):
    def __init__(
        self,
        y: Float[torch.Tensor, ""] | float | int = 0.0,
        z: Float[torch.Tensor, ""] | float | int = 0.0,
    ):
        super().__init__()
        self.func = Rotate3DKernel()
        self.y = to_tensor(y)
        self.z = to_tensor(z)

    def forward(
        self, dfk: HomMatrix3D, ifk: HomMatrix3D
    ) -> tuple[HomMatrix3D, HomMatrix3D]:
        return self.func.forward(dfk, ifk, self.y, self.z)


class MixedDimKinematic(KinematicElement):
    def __init__(self, module_2d: nn.Module, module_3d: nn.Module):
        super().__init__()
        self.module_2d = module_2d
        self.module_3d = module_3d

    def forward(self, dfk: HomMatrix, ifk: HomMatrix) -> tuple[HomMatrix, HomMatrix]:
        if dfk[0].shape[0] == 3:
            return self.module_2d(dfk, ifk)
        else:
            return self.module_3d(dfk, ifk)


class Gap(KinematicElement):
    """
    Translation along the X axis
    """

    def __init__(
        self,
        X: Float[torch.Tensor, ""] | float,
        trainable: bool = False,
    ):
        super().__init__()
        self.X = init_param(self, "X", X, trainable)
        self.func2d = Gap2DKernel()
        self.func3d = Gap3DKernel()

    def forward(self, dfk: HomMatrix, ifk: HomMatrix) -> tuple[HomMatrix, HomMatrix]:
        if dfk[0].shape[0] == 3:
            return self.func2d.forward(dfk, ifk, self.X)
        else:
            return self.func3d.forward(dfk, ifk, self.X)

    def reverse(self) -> Self:
        return type(self)(-self.X, self.trainable)


class Rotate(KinematicElement):
    def __init__(
        self, angles: tuple[float | int, float | int] | Float[torch.Tensor, "2"]
    ):
        super().__init__()
        self.mixed_dim = MixedDimKinematic(
            Rotate2D(angles[0]), Rotate3D(angles[1], angles[0])
        )

    def forward(self, dfk: HomMatrix, ifk: HomMatrix) -> tuple[HomMatrix, HomMatrix]:
        return self.mixed_dim(dfk, ifk)


class Translate(KinematicElement):
    def __init__(
        self,
        x: Float[torch.Tensor, ""] | float | int = 0.0,
        y: Float[torch.Tensor, ""] | float | int = 0.0,
        z: Float[torch.Tensor, ""] | float | int = 0.0,
    ):
        super().__init__()
        self.mixed_dim = MixedDimKinematic(Translate2D(x, y), Translate3D(x, y, z))

    def forward(self, dfk: HomMatrix, ifk: HomMatrix) -> tuple[HomMatrix, HomMatrix]:
        return self.mixed_dim(dfk, ifk)
