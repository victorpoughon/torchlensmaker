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
    Tf,
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
    inputs = {}
    params = {"X": ScalarTensor}
    outputs = {"joint": Tf}

    def apply(self, X: ScalarTensor) -> Tf:
        return hom_translate_2d(torch.stack((X, torch.zeros_like(X))))

    def example_inputs(self, dtype: torch.dtype, device: torch.device) -> tuple[Tf]:
        return tuple()

    def example_params(
        self, dtype: torch.dtype, device: torch.device
    ) -> tuple[ScalarTensor]:
        return (torch.tensor(5.0, dtype=dtype, device=device),)


class Gap3DKernel(FunctionalKernel):
    inputs = {}
    params = {"X": ScalarTensor}
    outputs = {"joint": Tf}

    def apply(self, X: ScalarTensor) -> Tf:
        return hom_translate_3d(
            torch.stack((X, torch.zeros_like(X), torch.zeros_like(X)))
        )

    def example_inputs(self, dtype: torch.dtype, device: torch.device) -> tuple[Tf]:
        return tuple()

    def example_params(
        self, dtype: torch.dtype, device: torch.device
    ) -> tuple[ScalarTensor]:
        return (torch.tensor(1.0, dtype=dtype, device=device),)


class Translate2DKernel(FunctionalKernel):
    inputs = {}
    params = {"X": ScalarTensor, "Y": ScalarTensor}
    outputs = {"joint": Tf}

    def apply(self, X: ScalarTensor, Y: ScalarTensor) -> Tf:
        return hom_translate_2d(torch.stack((X, Y)))

    def example_inputs(self, dtype: torch.dtype, device: torch.device) -> tuple[Tf]:
        return tuple()

    def example_params(
        self, dtype: torch.dtype, device: torch.device
    ) -> tuple[ScalarTensor, ScalarTensor]:
        return (
            torch.tensor(5.0, dtype=dtype, device=device),
            torch.tensor(2.0, dtype=dtype, device=device),
        )


class Translate3DKernel(FunctionalKernel):
    inputs = {}
    params = {"X": ScalarTensor, "Y": ScalarTensor, "Z": ScalarTensor}
    outputs = {"joint": Tf}

    def apply(self, X: ScalarTensor, Y: ScalarTensor, Z: ScalarTensor) -> Tf:
        return hom_translate_3d(torch.stack((X, Y, Z)))

    def example_inputs(self, dtype: torch.dtype, device: torch.device) -> tuple[Tf]:
        return tuple()

    def example_params(
        self, dtype: torch.dtype, device: torch.device
    ) -> tuple[ScalarTensor, ScalarTensor, ScalarTensor]:
        return (
            torch.tensor(1.0, dtype=dtype, device=device),
            torch.tensor(2.0, dtype=dtype, device=device),
            torch.tensor(3.0, dtype=dtype, device=device),
        )


class Rotate2DKernel(FunctionalKernel):
    "2D rotation in degrees"

    inputs = {}
    params = {"theta": ScalarTensor}
    outputs = {"joint": Tf}

    def apply(self, theta: ScalarTensor) -> Tf:
        return hom_rotate_2d(torch.deg2rad(theta))

    def example_inputs(self, dtype: torch.dtype, device: torch.device) -> tuple[Tf]:
        return tuple()

    def example_params(
        self, dtype: torch.dtype, device: torch.device
    ) -> tuple[ScalarTensor]:
        return (torch.tensor(0.1, dtype=dtype, device=device),)


class Rotate3DKernel(FunctionalKernel):
    "3D rotation in degrees"

    inputs = {}
    params = {"y": ScalarTensor, "z": ScalarTensor}
    outputs = {"joint": Tf}

    def apply(self, y: ScalarTensor, z: ScalarTensor) -> Tf:
        return hom_rotate_3d(y, z)

    def example_inputs(self, dtype: torch.dtype, device: torch.device) -> tuple[Tf]:
        return tuple()

    def example_params(
        self, dtype: torch.dtype, device: torch.device
    ) -> tuple[ScalarTensor, ScalarTensor]:
        return (
            torch.tensor(0.1, dtype=dtype, device=device),
            torch.tensor(0.2, dtype=dtype, device=device),
        )


class KinematicChainAppend2DKernel(FunctionalKernel):
    inputs = {"fk_in": Tf, "joint": Tf}
    params = {}
    outputs = {"fk_out": Tf}

    def apply(self, fk: Tf, joint: Tf) -> Tf:
        return kinematic_chain_append_2d(fk, joint)

    def example_inputs(self, dtype: torch.dtype, device: torch.device) -> tuple[Tf]:
        return (
            hom_identity_2d(dtype=dtype, device=device),
            hom_identity_2d(dtype=dtype, device=device),
        )

    def example_params(
        self, dtype: torch.dtype, device: torch.device
    ) -> tuple[ScalarTensor, ScalarTensor]:
        return tuple()


class KinematicChainAppend3DKernel(FunctionalKernel):
    inputs = {"fk_in": Tf, "joint": Tf}
    params = {}
    outputs = {"fk_out": Tf}

    def apply(self, fk: Tf, joint: Tf) -> Tf:
        return kinematic_chain_append_3d(fk, joint)

    def example_inputs(self, dtype: torch.dtype, device: torch.device) -> tuple[Tf]:
        return (
            hom_identity_3d(dtype=dtype, device=device),
            hom_identity_3d(dtype=dtype, device=device),
        )

    def example_params(
        self, dtype: torch.dtype, device: torch.device
    ) -> tuple[ScalarTensor, ScalarTensor]:
        return tuple()
