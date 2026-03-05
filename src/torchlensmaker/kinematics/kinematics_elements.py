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
    HomMatrix,
    ScalarTensor,
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

from torchlensmaker.core.base_module import MultiForwardModule, multiforward


class KinematicElement(MultiForwardModule):
    @multiforward
    def sequential(self, data: OpticalData) -> OpticalData:
        fk = self.kinematic_prograde(data.fk)
        return data.replace(fk=fk)

    @multiforward
    def sequential_retrograde(self, data: OpticalData) -> OpticalData:
        fk = self.kinematic_retrograde(data.fk)
        return data.replace(fk=fk)


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

    def clone(self, **overrides: Any) -> Self:
        kwargs = dict(x=self.x, trainable=self.x.requires_grad)
        return type(self)(**kwargs | overrides)

    def __repr__(self) -> str:
        return f"{self._get_name()}(x={self.x.item()})"

    @multiforward
    def kinematic_prograde(self, fk: Tf) -> Tf:
        func = self.func2d if fk.pdim() == 2 else self.func3d
        return func.apply(fk, self.x)

    @multiforward
    def kinematic_retrograde(self, fk: Tf) -> Tf:
        func = self.func2d if fk.pdim() == 2 else self.func3d
        return func.apply(fk, -self.x)


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

    def clone(self, **overrides: Any) -> Self:
        kwargs = dict(
            x=self.x,
            y=self.y,
            trainable=(self.x.requires_grad, self.y.requires_grad),
        )
        return type(self)(**kwargs | overrides)

    @multiforward
    def kinematic_prograde(self, fk: Tf) -> Tf:
        return self.func.apply(fk, self.x, self.y)

    @multiforward
    def kinematic_retrograde(self, fk: Tf) -> Tf:
        return self.func.apply(fk, -self.x, -self.y)


class TranslateVec2D(KinematicElement):
    def __init__(
        self,
        t: list[float] | Float[torch.Tensor, "2"] | nn.Parameter,
        trainable: bool = False,
    ):
        super().__init__()
        self.func = Translate2DKernel()
        self.t = init_param(self, "t", t, trainable)

    def clone(self, **overrides: Any) -> Self:
        kwargs = dict(
            t=self.t,
            trainable=self.t.requires_grad,
        )
        return type(self)(**kwargs | overrides)

    @multiforward
    def kinematic_prograde(self, fk: Tf) -> Tf:
        return self.func.apply(fk, *torch.unbind(self.t))

    @multiforward
    def kinematic_retrograde(self, fk: Tf) -> Tf:
        return self.func.apply(fk, *torch.unbind(-self.t))


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

    def clone(self, **overrides: Any) -> Self:
        kwargs = dict(
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

    @multiforward
    def kinematic_prograde(self, fk: Tf) -> Tf:
        return self.func.apply(fk, self.x, self.y, self.z)

    @multiforward
    def kinematic_retrograde(self, fk: Tf) -> Tf:
        return self.func.apply(fk, -self.x, -self.y, -self.z)


class TranslateVec3D(KinematicElement):
    def __init__(
        self,
        t: list[float] | Float[torch.Tensor, "3"] | nn.Parameter,
        trainable: bool = False,
    ):
        super().__init__()
        self.func = Translate3DKernel()
        self.t = init_param(self, "t", t, trainable)

    def clone(self, **overrides: Any) -> Self:
        kwargs = dict(
            t=self.t,
            trainable=self.t.requires_grad,
        )
        return type(self)(**kwargs | overrides)

    @multiforward
    def kinematic_prograde(self, fk: Tf) -> Tf:
        return self.func.apply(fk, *torch.unbind(self.t))

    @multiforward
    def kinematic_retrograde(self, fk: Tf) -> Tf:
        return self.func.apply(fk, *torch.unbind(-self.t))


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

    def clone(self, **overrides: Any) -> Self:
        kwargs = dict(
            theta=self.theta,
            trainable=self.theta.requires_grad,
        )
        return type(self)(**kwargs | overrides)

    @multiforward
    def kinematic_prograde(self, fk: Tf) -> Tf:
        return self.func.apply(fk, self.theta)

    @multiforward
    def kinematic_retrograde(self, fk: Tf) -> Tf:
        return self.func.apply(fk, -self.theta)


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

    def clone(self, **overrides: Any) -> Self:
        kwargs = dict(
            x=self.x,
            y=self.y,
            trainable=(self.x.requires_grad, self.y.requires_grad),
        )
        return type(self)(**kwargs | overrides)

    @multiforward
    def kinematic_prograde(self, fk: Tf) -> Tf:
        return self.func.apply(fk, self.x, self.y)

    @multiforward
    def kinematic_retrograde(self, fk: Tf) -> Tf:
        raise RuntimeError(
            "AbsolutePosition2D(): element is not reversable (cannot evaluate retrograde)"
        )


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

    def clone(self, **overrides: Any) -> Self:
        kwargs = dict(
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

    @multiforward
    def kinematic_prograde(self, fk: Tf) -> Tf:
        return self.func.apply(fk, self.x, self.y, self.z)

    @multiforward
    def kinematic_retrograde(self, fk: Tf) -> Tf:
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

    def clone(self, **overrides: Any) -> Self:
        kwargs = dict(
            y=self.y,
            z=self.z,
            trainable=(
                self.y.requires_grad,
                self.z.requires_grad,
            ),
        )
        return type(self)(**kwargs | overrides)

    @multiforward
    def kinematic_prograde(self, fk: Tf) -> Tf:
        return self.func.apply(fk, self.y, self.z)

    @multiforward
    def kinematic_retrograde(self, fk: Tf) -> Tf:
        # TODO
        raise NotImplementedError("TODO")


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

    def clone(self, **overrides: Any) -> Self:
        kwargs = dict(
            angles=torch.stack((self.z, self.y)),
            trainable=(self.z.requires_grad, self.y.requires_grad),
        )
        return type(self)(**kwargs | overrides)

    @multiforward
    def kinematic_prograde(self, fk: Tf) -> Tf:
        if fk.shape[0] == 3:
            return self.func2d.apply(fk, self.z)
        else:
            return self.func3d.apply(fk, self.y, self.z)

    @multiforward
    def kinematic_retrograde(self, fk: Tf) -> Tf:
        # TODO
        raise NotImplementedError("TODO")


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

    def clone(self, **overrides: Any) -> Self:
        kwargs = dict(
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

    @multiforward
    def kinematic_prograde(self, fk: Tf) -> Tf:
        if fk.shape[0] == 3:
            return self.func2d.apply(fk, self.x, self.y)
        else:
            return self.func3d.apply(fk, self.x, self.y, self.z)

    @multiforward
    def kinematic_retrograde(self, fk: Tf) -> Tf:
        if fk.shape[0] == 3:
            return self.func2d.apply(fk, self.x, self.y)
        else:
            return self.func3d.apply(fk, -self.x, -self.y, -self.z)
