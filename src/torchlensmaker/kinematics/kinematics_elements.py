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

import torch
import torch.nn as nn
from jaxtyping import Float

from torchlensmaker.core.tensor_manip import expand_bool_tuple, init_param, to_tensor
from torchlensmaker.elements.sequential_data import SequentialData
from torchlensmaker.elements.sequential_element import SequentialElement
from torchlensmaker.types import (
    Direction,
    HomMatrix,
    ScalarTensor,
    Tf,
)

from .kinematics_kernels import (
    Gap2DKernel,
    Gap3DKernel,
    KinematicChainAppend2DKernel,
    KinematicChainAppend3DKernel,
    Rotate2DKernel,
    Rotate3DKernel,
    Translate2DKernel,
    Translate3DKernel,
)


class KinematicElement(SequentialElement):
    def sequential(self, data: SequentialData) -> SequentialData:
        fk = self(data.fk, data.direction)
        return data.replace(fk=fk)

    def forward(self, fk: Tf, direction: Direction) -> Tf:
        raise NotImplementedError


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
        self.kernel_joint2d = Gap2DKernel()
        self.kernel_joint3d = Gap3DKernel()
        self.kernel_fk2d = KinematicChainAppend2DKernel()
        self.kernel_fk3d = KinematicChainAppend3DKernel()

    def clone(self, **overrides: Any) -> Self:
        kwargs: dict[str, Any] = dict(x=self.x, trainable=self.x.requires_grad)
        return type(self)(**kwargs | overrides)

    def __repr__(self) -> str:
        return f"{self._get_name()}(x={self.x.item()})"

    def forward(self, fk: Tf, direction: Direction) -> Tf:
        kernel_joint = self.kernel_joint2d if fk.pdim() == 2 else self.kernel_joint3d
        kernel_fk = self.kernel_fk2d if fk.pdim() == 2 else self.kernel_fk3d
        joint = kernel_joint.apply(self.x)

        if direction.is_retrograde():
            joint = joint.flip()

        return kernel_fk.apply(fk, joint)


class Translate2D(KinematicElement):
    def __init__(
        self,
        x: float | ScalarTensor | nn.Parameter = 0.0,
        y: float | ScalarTensor | nn.Parameter = 0.0,
        trainable: bool | tuple[bool, ...] = False,
    ):
        super().__init__()
        xt, yt = expand_bool_tuple(2, trainable)
        self.x = init_param(self, "x", x, xt)
        self.y = init_param(self, "y", y, yt)
        self.kernel_joint = Translate2DKernel()
        self.kernel_fk2d = KinematicChainAppend2DKernel()

    def clone(self, **overrides: Any) -> Self:
        kwargs: dict[str, Any] = dict(
            x=self.x,
            y=self.y,
            trainable=(self.x.requires_grad, self.y.requires_grad),
        )
        return type(self)(**kwargs | overrides)

    def forward(self, fk: Tf, direction: Direction) -> Tf:
        joint = self.kernel_joint.apply(self.x, self.y)

        if direction.is_retrograde():
            joint = joint.flip()

        return self.kernel_fk2d.apply(fk, joint)


class TranslateVec2D(KinematicElement):
    def __init__(
        self,
        t: list[float] | Float[torch.Tensor, "2"] | nn.Parameter,
        trainable: bool = False,
    ):
        super().__init__()
        self.t = init_param(self, "t", t, trainable)
        self.kernel_joint = Translate2DKernel()
        self.kernel_fk2d = KinematicChainAppend2DKernel()

    def clone(self, **overrides: Any) -> Self:
        kwargs: dict[str, Any] = dict(
            t=self.t,
            trainable=self.t.requires_grad,
        )
        return type(self)(**kwargs | overrides)

    def forward(self, fk: Tf, direction: Direction) -> Tf:
        joint = self.kernel_joint.apply(*torch.unbind(self.t))

        if direction.is_retrograde():
            joint = joint.flip()

        return self.kernel_fk2d.apply(fk, joint)


class Translate3D(KinematicElement):
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
        self.kernel_joint = Translate3DKernel()
        self.kernel_fk3d = KinematicChainAppend3DKernel()

    def clone(self, **overrides: Any) -> Self:
        kwargs: dict[str, Any] = dict(
            x=self.x,
            y=self.y,
            z=self.z,
            trainable=(
                self.x.requires_grad,
                self.y.requires_grad,
                self.z.requires_grad,
            ),
        )
        return type(self)(**kwargs | overrides)

    def forward(self, fk: Tf, direction: Direction) -> Tf:
        joint = self.kernel_joint.apply(self.x, self.y, self.z)

        if direction.is_retrograde():
            joint = joint.flip()

        return self.kernel_fk3d.apply(fk, joint)


class TranslateVec3D(KinematicElement):
    def __init__(
        self,
        t: list[float] | Float[torch.Tensor, "3"] | nn.Parameter,
        trainable: bool = False,
    ):
        super().__init__()
        self.t = init_param(self, "t", t, trainable)
        self.kernel_joint = Translate3DKernel()
        self.kernel_fk3d = KinematicChainAppend3DKernel()

    def clone(self, **overrides: Any) -> Self:
        kwargs: dict[str, Any] = dict(
            t=self.t,
            trainable=self.t.requires_grad,
        )
        return type(self)(**kwargs | overrides)

    def forward(self, fk: Tf, direction: Direction) -> Tf:
        joint = self.kernel_joint.apply(*torch.unbind(self.t))

        if direction.is_retrograde():
            joint = joint.flip()

        return self.kernel_fk3d.apply(fk, joint)


class Rotate2D(KinematicElement):
    "2D rotation (in degrees)"

    def __init__(
        self,
        theta: float | ScalarTensor | nn.Parameter = 0.0,
        trainable: bool = False,
    ):
        super().__init__()
        self.theta = init_param(self, "theta", theta, trainable)
        self.kernel_joint = Rotate2DKernel()
        self.kernel_fk2d = KinematicChainAppend2DKernel()

    def clone(self, **overrides: Any) -> Self:
        kwargs: dict[str, Any] = dict(
            theta=self.theta,
            trainable=self.theta.requires_grad,
        )
        return type(self)(**kwargs | overrides)

    def forward(self, fk: Tf, direction: Direction) -> Tf:
        joint = self.kernel_joint.apply(self.theta)

        if direction.is_retrograde():
            joint = joint.flip()

        return self.kernel_fk2d.apply(fk, joint)


class AbsolutePosition2D(KinematicElement):
    def __init__(
        self,
        x: float | ScalarTensor | nn.Parameter = 0.0,
        y: float | ScalarTensor | nn.Parameter = 0.0,
        trainable: bool | tuple[bool, ...] = False,
    ):
        super().__init__()
        self.kernel_joint = Translate2DKernel()
        xt, yt = expand_bool_tuple(2, trainable)
        self.x = init_param(self, "x", x, xt)
        self.y = init_param(self, "y", y, yt)

    def clone(self, **overrides: Any) -> Self:
        kwargs: dict[str, Any] = dict(
            x=self.x,
            y=self.y,
            trainable=(self.x.requires_grad, self.y.requires_grad),
        )
        return type(self)(**kwargs | overrides)

    def forward(self, fk: Tf, direction: Direction) -> Tf:
        if direction.is_retrograde():
            raise RuntimeError(
                "AbsolutePosition2D cannot be evaluated in reverse (retrograde) direction"
            )

        return self.kernel_joint.apply(self.x, self.y)


class AbsolutePosition3D(KinematicElement):
    def __init__(
        self,
        x: float | ScalarTensor | nn.Parameter = 0.0,
        y: float | ScalarTensor | nn.Parameter = 0.0,
        z: float | ScalarTensor | nn.Parameter = 0.0,
        trainable: bool | tuple[bool, ...] = False,
    ):
        super().__init__()
        self.kernel_joint = Translate3DKernel()
        xt, yt, zt = expand_bool_tuple(3, trainable)
        self.x = init_param(self, "x", x, xt)
        self.y = init_param(self, "y", y, yt)
        self.z = init_param(self, "z", z, zt)

    def clone(self, **overrides: Any) -> Self:
        kwargs: dict[str, Any] = dict(
            x=self.x,
            y=self.y,
            z=self.z,
            trainable=(
                self.x.requires_grad,
                self.y.requires_grad,
                self.z.requires_grad,
            ),
        )
        return type(self)(**kwargs | overrides)

    def forward(self, fk: Tf, direction: Direction) -> Tf:
        if direction.is_retrograde():
            raise RuntimeError(
                "AbsolutePosition3D cannot be evaluated in reverse (retrograde) direction"
            )

        return self.kernel_joint.apply(self.x, self.y, self.z)


class Rotate3D(KinematicElement):
    def __init__(
        self,
        y: float | ScalarTensor | nn.Parameter = 0.0,
        z: float | ScalarTensor | nn.Parameter = 0.0,
        trainable: bool | tuple[bool, ...] = False,
    ):
        super().__init__()
        self.kernel_joint = Rotate3DKernel()
        self.kernel_fk3d = KinematicChainAppend3DKernel()
        yt, zt = expand_bool_tuple(2, trainable)
        self.y = init_param(self, "y", y, yt)
        self.z = init_param(self, "z", z, zt)

    def clone(self, **overrides: Any) -> Self:
        kwargs: dict[str, Any] = dict(
            y=self.y,
            z=self.z,
            trainable=(
                self.y.requires_grad,
                self.z.requires_grad,
            ),
        )
        return type(self)(**kwargs | overrides)

    def forward(self, fk: Tf, direction: Direction) -> Tf:
        joint = self.kernel_joint.apply(self.y, self.z)

        if direction.is_retrograde():
            joint = joint.flip()

        return self.kernel_fk3d.apply(fk, joint)


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
        self.kernel_joint2d = Rotate2DKernel()
        self.kernel_joint3d = Rotate3DKernel()
        self.kernel_fk2d = KinematicChainAppend2DKernel()
        self.kernel_fk3d = KinematicChainAppend3DKernel()

    def clone(self, **overrides: Any) -> Self:
        kwargs: dict[str, Any] = dict(
            angles=torch.stack((self.z, self.y)),
            trainable=(self.z.requires_grad, self.y.requires_grad),
        )
        return type(self)(**kwargs | overrides)

    def forward(self, fk: Tf, direction: Direction) -> Tf:
        if fk.pdim() == 2:
            joint = self.kernel_joint2d.apply(self.z)
            kernel_fk = self.kernel_fk2d
        else:
            joint = self.kernel_joint3d.apply(self.y, self.z)
            kernel_fk = self.kernel_fk3d

        if direction.is_retrograde():
            joint = joint.flip()

        return kernel_fk.apply(fk, joint)


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
        self.kernel_joint2d = Translate2DKernel()
        self.kernel_joint3d = Translate3DKernel()
        self.kernel_fk2d = KinematicChainAppend2DKernel()
        self.kernel_fk3d = KinematicChainAppend3DKernel()

    def clone(self, **overrides: Any) -> Self:
        kwargs: dict[str, Any] = dict(
            x=self.x,
            y=self.y,
            z=self.z,
            trainable=(
                self.x.requires_grad,
                self.y.requires_grad,
                self.z.requires_grad,
            ),
        )
        return type(self)(**kwargs | overrides)

    def forward(self, fk: Tf, direction: Direction) -> Tf:
        if fk.pdim() == 2:
            joint = self.kernel_joint2d.apply(self.x, self.y)
            kernel_fk = self.kernel_fk2d
        else:
            joint = self.kernel_joint3d.apply(self.x, self.y, self.z)
            kernel_fk = self.kernel_fk3d

        if direction.is_retrograde():
            joint = joint.flip()

        return kernel_fk.apply(fk, joint)
