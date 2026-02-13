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

from torchlensmaker.core.tensor_manip import to_tensor, init_param, expand_bool_tuple
from torchlensmaker.optical_data import OpticalData

from torchlensmaker.types import (
    HomMatrix2D,
    HomMatrix3D,
    HomMatrix,
    ScalarTensor,
    Tf2D,
    Tf3D,
    Tf,
)
from .kinematics_kernels import (
    AbsolutePosition2DKernel,
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
        fk = self(data.fk)
        return data.replace(fk=fk)


class KinematicSequential(nn.Module):
    def __init__(self, *sequence: nn.Module):
        super().__init__()
        self.sequence = nn.ModuleList(sequence)

    def forward(self, fk: Tf):
        for mod in self.sequence:
            fk = mod(fk)
        return fk


# TODO this is used for lens only to make a "kinematic only" sequential model
# no kernel needed for now
class ExactKinematicElement2D(KinematicElement):
    def __init__(self, joint: Tf2D):
        super().__init__()
        self.joint = joint

    def forward(self, fk: Tf2D) -> Tf2D:
        return kinematic_chain_append(fk, self.joint)

    def reverse(self) -> Self:
        return type(self)(Tf2D(self.joint.inverse, self.joint.direct))


class Gap(KinematicElement):
    """
    Translation along the X axis
    Works in both 2D and 3D and shares a single parameter x.
    """

    def __init__(
        self,
        x: float | ScalarTensor | nn.Parameter,
        trainable: bool = False,
    ):
        super().__init__()
        self.x = init_param(self, "x", x, trainable)
        self.func2d = Gap2DKernel()
        self.func3d = Gap3DKernel()

    def forward(self, fk: Tf) -> Tf:
        if fk.shape[0] == 3:
            return self.func2d.forward(fk, self.x)
        else:
            return self.func3d.forward(fk, self.x)

    def reverse(self) -> Self:
        return type(self)(-self.x.detach(), self.x.requires_grad)


class Translate2D(KinematicElement):
    def __init__(
        self,
        x: float | ScalarTensor | nn.Parameter = 0.0,
        y: float | ScalarTensor | nn.Parameter = 0.0,
        trainable: bool | tuple[bool, ...] = False,
    ):
        super().__init__()
        self.func = Translate2DKernel()
        xt, yt = expand_bool_tuple(2, trainable)
        self.x = init_param(self, "x", x, xt)
        self.y = init_param(self, "y", y, yt)

    def forward(self, fk: Tf2D) -> Tf2D:
        return self.func.forward(fk, self.x, self.y)

    def reverse(self) -> Self:
        return type(self)(
            -self.x.detach(),
            -self.y.detach(),
            (self.x.requires_grad, self.y.requires_grad),
        )


class TranslateVec2D(KinematicElement):
    def __init__(
        self,
        t: list[float] | Float[torch.Tensor, "2"] | nn.Parameter,
        trainable: bool = False,
    ):
        super().__init__()
        self.func = Translate2DKernel()
        self.t = init_param(self, "t", t, trainable)

    def forward(self, fk: Tf2D) -> Tf2D:
        return self.func.forward(fk, *torch.unbind(self.t))

    def reverse(self) -> Self:
        return type(self)(-self.t.detach(), self.t.requires_grad)


class Translate3D(KinematicElement):
    def __init__(
        self,
        x: float | ScalarTensor | nn.Parameter = 0.0,
        y: float | ScalarTensor | nn.Parameter = 0.0,
        z: float | ScalarTensor | nn.Parameter = 0.0,
        trainable: bool | tuple[bool, ...] = False,
    ):
        super().__init__()
        self.func = Translate3DKernel()
        xt, yt, zt = expand_bool_tuple(3, trainable)
        self.x = init_param(self, "x", x, xt)
        self.y = init_param(self, "y", y, yt)
        self.z = init_param(self, "z", z, zt)

    def forward(self, fk: Tf3D) -> Tf3D:
        return self.func.forward(fk, self.x, self.y, self.z)

    def reverse(self) -> Self:
        return type(self)(
            -self.x.detach(),
            -self.y.detach(),
            -self.z.detach(),
            (self.x.requires_grad, self.y.requires_grad, self.z.requires_grad),
        )


class TranslateVec3D(KinematicElement):
    def __init__(
        self,
        t: list[float] | Float[torch.Tensor, "3"] | nn.Parameter,
        trainable: bool = False,
    ):
        super().__init__()
        self.func = Translate3DKernel()
        self.t = init_param(self, "t", t, trainable)

    def forward(self, fk: Tf3D) -> Tf3D:
        return self.func.forward(fk, *torch.unbind(self.t))

    def reverse(self) -> Self:
        return type(self)(-self.t.detach(), self.t.requires_grad)


class Rotate2D(KinematicElement):
    "2D rotation (in degrees)"

    def __init__(
        self,
        theta: float | ScalarTensor | nn.Parameter = 0.0,
        trainable: bool = False,
    ):
        super().__init__()
        self.func = Rotate2DKernel()
        self.theta = init_param(self, "theta", theta, trainable)

    def forward(self, fk: Tf2D) -> Tf2D:
        return self.func.forward(fk, self.theta)

    def reverse(self) -> Self:
        return type(self)(-self.theta.detach(), self.theta.requires_grad)


class AbsolutePosition2D(KinematicElement):
    def __init__(
        self,
        x: float | ScalarTensor | nn.Parameter = 0.0,
        y: float | ScalarTensor | nn.Parameter = 0.0,
        trainable: bool | tuple[bool, ...] = False,
    ):
        super().__init__()
        self.func = AbsolutePosition2DKernel()
        xt, yt = expand_bool_tuple(2, trainable)
        self.x = init_param(self, "x", x, xt)
        self.y = init_param(self, "y", y, yt)

    def forward(self, fk: Tf2D) -> Tf2D:
        return self.func.forward(fk, self.x, self.y)

    def reverse(self) -> Self:
        raise RuntimeError("AbsolutePosition2D kinematic element is not reversable")


class AbsolutePosition3D(KinematicElement):
    def __init__(
        self,
        x: float | ScalarTensor | nn.Parameter = 0.0,
        y: float | ScalarTensor | nn.Parameter = 0.0,
        z: float | ScalarTensor | nn.Parameter = 0.0,
        trainable: bool | tuple[bool, ...] = False,
    ):
        super().__init__()
        self.func = AbsolutePosition3DKernel()
        xt, yt, zt = expand_bool_tuple(3, trainable)
        self.x = init_param(self, "x", x, xt)
        self.y = init_param(self, "y", y, yt)
        self.z = init_param(self, "z", z, zt)

    def forward(self, fk: Tf3D) -> Tf3D:
        return self.func.forward(fk, self.x, self.y, self.z)

    def reverse(self) -> Self:
        raise RuntimeError("AbsolutePosition3D kinematic element is not reversable")


class Rotate3D(KinematicElement):
    def __init__(
        self,
        y: float | ScalarTensor | nn.Parameter = 0.0,
        z: float | ScalarTensor | nn.Parameter = 0.0,
        trainable: bool | tuple[bool, ...] = False,
    ):
        super().__init__()
        self.func = Rotate3DKernel()
        yt, zt = expand_bool_tuple(2, trainable)
        self.y = init_param(self, "y", y, yt)
        self.z = init_param(self, "z", z, zt)

    def forward(self, fk: Tf3D) -> Tf3D:
        return self.func.forward(fk, self.y, self.z)

    # TODO support reverse for 3D rotations


class Rotate(KinematicElement):
    """
    Mixed dimension rotation
    """

    def __init__(
        self,
        angles: tuple[float] | Float[torch.Tensor, "2"] | nn.Parameter,
        trainable: bool | tuple[bool, ...] = False,
    ):
        super().__init__()
        zt, yt = expand_bool_tuple(2, trainable)
        z, y = to_tensor(angles).unbind()
        self.z = init_param(self, "z", z, zt)
        self.y = init_param(self, "y", y, yt)
        self.func2d = Rotate2DKernel()
        self.func3d = Rotate3DKernel()

    def forward(self, fk: Tf) -> Tf:
        if fk.shape[0] == 3:
            return self.func2d.forward(fk, self.z)
        else:
            return self.func3d.forward(fk, self.y, self.z)


class Translate(KinematicElement):
    """
    Mixed dimension translation
    """

    def __init__(
        self,
        x: float | ScalarTensor | nn.Parameter = 0.0,
        y: float | ScalarTensor | nn.Parameter = 0.0,
        z: float | ScalarTensor | nn.Parameter = 0.0,
        trainable: bool | tuple[bool, ...] = False,
    ):
        super().__init__()
        xt, yt, zt = expand_bool_tuple(3, trainable)
        self.x = init_param(self, "x", x, xt)
        self.y = init_param(self, "y", y, yt)
        self.z = init_param(self, "z", z, zt)
        self.func2d = Translate2DKernel()
        self.func3d = Translate3DKernel()

    def forward(self, fk: Tf) -> Tf:
        if fk.shape[0] == 3:
            return self.func2d.forward(fk, self.x, self.y)
        else:
            return self.func3d.forward(fk, self.x, self.y, self.z)
