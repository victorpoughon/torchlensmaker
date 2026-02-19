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

from torchlensmaker.types import (
    ScalarTensor,
    Tf2D,
    Tf3D,
)

from .homogeneous_geometry import (
    hom_rotate_2d,
    hom_rotate_3d,
    hom_translate_2d,
    hom_translate_3d,
    hom_identity_2d,
    hom_identity_3d,
    kinematic_chain_append_2d,
    kinematic_chain_append_3d,
)
from torchlensmaker.core.functional_kernel import FunctionalKernel


class Gap2DKernel(FunctionalKernel):
    inputs = {"tf_in": Tf2D}
    params = {"X": ScalarTensor}
    outputs = {"tf_out": Tf2D}

    @staticmethod
    def apply(fk: Tf2D, X: ScalarTensor) -> Tf2D:
        joint = hom_translate_2d(torch.stack((X, torch.zeros_like(X))))
        return kinematic_chain_append_2d(fk, joint)

    @staticmethod
    def example_inputs(dtype: torch.dtype, device: torch.device) -> tuple[Tf2D]:
        return (hom_identity_2d(dtype=dtype, device=device),)

    @staticmethod
    def example_params(dtype: torch.dtype, device: torch.device) -> tuple[ScalarTensor]:
        return (torch.tensor(5.0, dtype=dtype, device=device),)


class Gap3DKernel(FunctionalKernel):
    inputs = {"tf_in": Tf3D}
    params = {"X": ScalarTensor}
    outputs = {"tf_out": Tf3D}

    @staticmethod
    def apply(fk: Tf3D, X: ScalarTensor) -> Tf3D:
        joint = hom_translate_3d(
            torch.stack((X, torch.zeros_like(X), torch.zeros_like(X)))
        )
        return kinematic_chain_append_3d(fk, joint)

    @staticmethod
    def example_inputs(dtype: torch.dtype, device: torch.device) -> tuple[Tf3D]:
        return (hom_identity_3d(dtype=dtype, device=device),)

    @staticmethod
    def example_params(dtype: torch.dtype, device: torch.device) -> tuple[ScalarTensor]:
        return (torch.tensor(1.0, dtype=dtype, device=device),)


class Translate2DKernel(FunctionalKernel):
    inputs = {"tf_in": Tf2D}
    params = {"X": ScalarTensor, "Y": ScalarTensor}
    outputs = {"tf_out": Tf2D}

    @staticmethod
    def apply(fk: Tf2D, X: ScalarTensor, Y: ScalarTensor) -> Tf2D:
        joint = hom_translate_2d(torch.stack((X, Y)))
        return kinematic_chain_append_2d(fk, joint)

    @staticmethod
    def example_inputs(dtype: torch.dtype, device: torch.device) -> tuple[Tf2D]:
        return (hom_identity_2d(dtype=dtype, device=device),)

    @staticmethod
    def example_params(
        dtype: torch.dtype, device: torch.device
    ) -> tuple[ScalarTensor, ScalarTensor]:
        return (
            torch.tensor(5.0, dtype=dtype, device=device),
            torch.tensor(2.0, dtype=dtype, device=device),
        )


class Translate3DKernel(FunctionalKernel):
    inputs = {"tf_in": Tf3D}
    params = {"X": ScalarTensor, "Y": ScalarTensor, "Z": ScalarTensor}
    outputs = {"tf_out": Tf3D}

    @staticmethod
    def apply(fk: Tf3D, X: ScalarTensor, Y: ScalarTensor, Z: ScalarTensor) -> Tf3D:
        joint = hom_translate_3d(torch.stack((X, Y, Z)))
        return kinematic_chain_append_3d(fk, joint)

    @staticmethod
    def example_inputs(dtype: torch.dtype, device: torch.device) -> tuple[Tf3D]:
        return (hom_identity_3d(dtype=dtype, device=device),)

    @staticmethod
    def example_params(
        dtype: torch.dtype, device: torch.device
    ) -> tuple[ScalarTensor, ScalarTensor, ScalarTensor]:
        return (
            torch.tensor(1.0, dtype=dtype, device=device),
            torch.tensor(2.0, dtype=dtype, device=device),
            torch.tensor(3.0, dtype=dtype, device=device),
        )


class Rotate2DKernel(FunctionalKernel):
    "2D rotation in degrees"

    inputs = {"tf_in": Tf2D}
    params = {"theta": ScalarTensor}
    outputs = {"tf_out": Tf2D}

    @staticmethod
    def apply(fk: Tf2D, theta: ScalarTensor) -> Tf2D:
        joint = hom_rotate_2d(torch.deg2rad(theta))
        return kinematic_chain_append_2d(fk, joint)

    @staticmethod
    def example_inputs(dtype: torch.dtype, device: torch.device) -> tuple[Tf2D]:
        return (hom_identity_2d(dtype=dtype, device=device),)

    @staticmethod
    def example_params(dtype: torch.dtype, device: torch.device) -> tuple[ScalarTensor]:
        return (torch.tensor(0.1, dtype=dtype, device=device),)


class AbsolutePosition2DKernel(FunctionalKernel):
    inputs = {"tf_in": Tf2D}
    params = {"X": ScalarTensor, "Y": ScalarTensor}
    outputs = {"tf_out": Tf2D}

    @staticmethod
    def apply(fk: Tf2D, X: ScalarTensor, Y: ScalarTensor) -> Tf2D:
        return hom_translate_2d(torch.stack([X, Y]))

    @staticmethod
    def example_inputs(dtype: torch.dtype, device: torch.device) -> tuple[Tf2D]:
        return (hom_identity_2d(dtype=dtype, device=device),)

    @staticmethod
    def example_params(
        dtype: torch.dtype, device: torch.device
    ) -> tuple[ScalarTensor, ScalarTensor]:
        return (
            torch.tensor(5.0, dtype=dtype, device=device),
            torch.tensor(10.0, dtype=dtype, device=device),
        )


class AbsolutePosition3DKernel(FunctionalKernel):
    inputs = {"tf_in": Tf3D}
    params = {"X": ScalarTensor, "Y": ScalarTensor, "Z": ScalarTensor}
    outputs = {"tf_out": Tf3D}

    @staticmethod
    def apply(fk: Tf3D, X: ScalarTensor, Y: ScalarTensor, Z: ScalarTensor) -> Tf3D:
        return hom_translate_3d(torch.stack([X, Y, Z]))

    @staticmethod
    def example_inputs(dtype: torch.dtype, device: torch.device) -> tuple[Tf3D]:
        return (hom_identity_3d(dtype=dtype, device=device),)

    @staticmethod
    def example_params(
        dtype: torch.dtype, device: torch.device
    ) -> tuple[ScalarTensor, ScalarTensor, ScalarTensor]:
        return (
            torch.tensor(5.0, dtype=dtype, device=device),
            torch.tensor(10.0, dtype=dtype, device=device),
            torch.tensor(-15.0, dtype=dtype, device=device),
        )


class Rotate3DKernel(FunctionalKernel):
    "3D rotation in degrees"

    inputs = {"tf_in": Tf3D}
    params = {"y": ScalarTensor, "z": ScalarTensor}
    outputs = {"tf_out": Tf3D}

    @staticmethod
    def apply(fk: Tf3D, y: ScalarTensor, z: ScalarTensor) -> Tf3D:
        joint = hom_rotate_3d(y, z)
        return kinematic_chain_append_3d(fk, joint)

    @staticmethod
    def example_inputs(dtype: torch.dtype, device: torch.device) -> tuple[Tf3D]:
        return (hom_identity_3d(dtype=dtype, device=device),)

    @staticmethod
    def example_params(
        dtype: torch.dtype, device: torch.device
    ) -> tuple[ScalarTensor, ScalarTensor]:
        return (
            torch.tensor(0.1, dtype=dtype, device=device),
            torch.tensor(0.2, dtype=dtype, device=device),
        )
