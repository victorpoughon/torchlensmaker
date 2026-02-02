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
from jaxtyping import Float

from .homogeneous_geometry import (
    HomMatrix2D,
    HomMatrix3D,
    hom_rotate_2d,
    hom_rotate_3d,
    hom_translate_2d,
    hom_translate_3d,
    hom_identity_2d,
    hom_identity_3d,
    kinematic_chain_append,
)
from torchlensmaker.core.functional_kernel import FunctionalKernel


class Gap2DKernel(FunctionalKernel):
    input_names = ["dfk_in", "ifk_in"]
    param_names = ["X"]
    output_names = ["dfk_out", "ifk_out"]

    @staticmethod
    def forward(
        dfk: HomMatrix2D,
        ifk: HomMatrix2D,
        X: Float[torch.Tensor, ""],
    ) -> tuple[HomMatrix2D, HomMatrix2D]:
        hom, hom_inv = hom_translate_2d(torch.stack((X, torch.zeros_like(X))))
        return kinematic_chain_append(dfk, ifk, hom, hom_inv)

    @staticmethod
    def example_inputs(
        dtype: torch.dtype, device: torch.device
    ) -> tuple[HomMatrix2D, HomMatrix2D]:
        return hom_identity_2d(dtype=dtype, device=device)

    @staticmethod
    def example_params(
        dtype: torch.dtype, device: torch.device
    ) -> tuple[Float[torch.Tensor, ""]]:
        return (torch.tensor(5.0, dtype=dtype, device=device),)


class Gap3DKernel(FunctionalKernel):
    input_names = ["dfk_in", "ifk_in"]
    param_names = ["X"]
    output_names = ["dfk_out", "ifk_out"]

    @staticmethod
    def forward(
        dfk: HomMatrix3D,
        ifk: HomMatrix3D,
        X: Float[torch.Tensor, ""],
    ) -> tuple[HomMatrix3D, HomMatrix3D]:
        hom, hom_inv = hom_translate_3d(
            torch.stack((X, torch.zeros_like(X), torch.zeros_like(X)))
        )
        return kinematic_chain_append(dfk, ifk, hom, hom_inv)

    @staticmethod
    def example_inputs(
        dtype: torch.dtype, device: torch.device
    ) -> tuple[HomMatrix3D, HomMatrix3D]:
        return hom_identity_3d(dtype=dtype, device=device)

    @staticmethod
    def example_params(
        dtype: torch.dtype, device: torch.device
    ) -> tuple[Float[torch.Tensor, ""]]:
        return (torch.tensor(1.0, dtype=dtype, device=device),)


class Translate2DKernel(FunctionalKernel):
    input_names = ["dfk_in", "ifk_in"]
    param_names = ["X", "Y"]
    output_names = ["dfk_out", "ifk_out"]

    @staticmethod
    def forward(
        dfk: HomMatrix2D,
        ifk: HomMatrix2D,
        X: Float[torch.Tensor, ""],
        Y: Float[torch.Tensor, ""],
    ) -> tuple[HomMatrix2D, HomMatrix2D]:
        hom, hom_inv = hom_translate_2d(torch.stack((X, Y)))
        return kinematic_chain_append(dfk, ifk, hom, hom_inv)

    @staticmethod
    def example_inputs(
        dtype: torch.dtype, device: torch.device
    ) -> tuple[HomMatrix2D, HomMatrix2D]:
        return hom_identity_2d(dtype=dtype, device=device)

    @staticmethod
    def example_params(
        dtype: torch.dtype, device: torch.device
    ) -> tuple[Float[torch.Tensor, ""], Float[torch.Tensor, ""]]:
        return (
            torch.tensor(5.0, dtype=dtype, device=device),
            torch.tensor(2.0, dtype=dtype, device=device),
        )


class Translate3DKernel(FunctionalKernel):
    input_names = ["dfk_in", "ifk_in"]
    param_names = ["X", "Y", "Z"]
    output_names = ["dfk_out", "ifk_out"]

    @staticmethod
    def forward(
        dfk: HomMatrix2D,
        ifk: HomMatrix2D,
        X: Float[torch.Tensor, ""],
        Y: Float[torch.Tensor, ""],
        Z: Float[torch.Tensor, ""],
    ) -> tuple[HomMatrix3D, HomMatrix3D]:
        hom, hom_inv = hom_translate_3d(torch.stack((X, Y, Z)))
        return kinematic_chain_append(dfk, ifk, hom, hom_inv)

    @staticmethod
    def example_inputs(
        dtype: torch.dtype, device: torch.device
    ) -> tuple[HomMatrix3D, HomMatrix3D]:
        return hom_identity_3d(dtype=dtype, device=device)

    @staticmethod
    def example_params(
        dtype: torch.dtype, device: torch.device
    ) -> tuple[
        Float[torch.Tensor, ""], Float[torch.Tensor, ""], Float[torch.Tensor, ""]
    ]:
        return (
            torch.tensor(1.0, dtype=dtype, device=device),
            torch.tensor(2.0, dtype=dtype, device=device),
            torch.tensor(3.0, dtype=dtype, device=device),
        )


class Rotate2DKernel(FunctionalKernel):
    "2D rotation in degrees"

    input_names = ["dfk_in", "ifk_in"]
    param_names = ["theta"]
    output_names = ["dfk_out", "ifk_out"]

    @staticmethod
    def forward(
        dfk: HomMatrix2D, ifk: HomMatrix2D, theta: Float[torch.Tensor, ""]
    ) -> tuple[HomMatrix2D, HomMatrix2D]:
        hom, hom_inv = hom_rotate_2d(torch.deg2rad(theta))
        return kinematic_chain_append(dfk, ifk, hom, hom_inv)

    @staticmethod
    def example_inputs(
        dtype: torch.dtype, device: torch.device
    ) -> tuple[HomMatrix2D, HomMatrix2D]:
        return hom_identity_2d(dtype=dtype, device=device)

    @staticmethod
    def example_params(
        dtype: torch.dtype, device: torch.device
    ) -> tuple[Float[torch.Tensor, ""]]:
        return (torch.tensor(0.1, dtype=dtype, device=device),)


class AbsolutePosition2DKernel(FunctionalKernel):
    input_names = ["dfk_in", "ifk_in"]
    param_names = ["X", "Y"]
    output_names = ["dfk_out", "ifk_out"]

    @staticmethod
    def forward(
        dfk: HomMatrix2D,
        ifk: HomMatrix2D,
        X: Float[torch.Tensor, ""],
        Y: Float[torch.Tensor, ""],
    ) -> tuple[HomMatrix2D, HomMatrix2D]:
        return hom_translate_2d(torch.stack([X, Y]))

    @staticmethod
    def example_inputs(
        dtype: torch.dtype, device: torch.device
    ) -> tuple[HomMatrix2D, HomMatrix2D]:
        return hom_identity_2d(dtype=dtype, device=device)

    @staticmethod
    def example_params(
        dtype: torch.dtype, device: torch.device
    ) -> tuple[
        Float[torch.Tensor, ""],
        Float[torch.Tensor, ""],
    ]:
        return (
            torch.tensor(5.0, dtype=dtype, device=device),
            torch.tensor(10.0, dtype=dtype, device=device),
        )


class AbsolutePosition3DKernel(FunctionalKernel):
    input_names = ["dfk_in", "ifk_in"]
    param_names = ["X", "Y", "Z"]
    output_names = ["dfk_out", "ifk_out"]

    @staticmethod
    def forward(
        dfk: HomMatrix3D,
        ifk: HomMatrix3D,
        X: Float[torch.Tensor, ""],
        Y: Float[torch.Tensor, ""],
        Z: Float[torch.Tensor, ""],
    ) -> tuple[HomMatrix3D, HomMatrix3D]:
        return hom_translate_3d(torch.stack([X, Y, Z]))

    @staticmethod
    def example_inputs(
        dtype: torch.dtype, device: torch.device
    ) -> tuple[HomMatrix3D, HomMatrix3D]:
        return hom_identity_3d(dtype=dtype, device=device)

    @staticmethod
    def example_params(
        dtype: torch.dtype, device: torch.device
    ) -> tuple[
        Float[torch.Tensor, ""],
        Float[torch.Tensor, ""],
        Float[torch.Tensor, ""],
    ]:
        return (
            torch.tensor(5.0, dtype=dtype, device=device),
            torch.tensor(10.0, dtype=dtype, device=device),
            torch.tensor(-15.0, dtype=dtype, device=device),
        )


class Rotate3DKernel(FunctionalKernel):
    "3D rotation in degrees"

    input_names = ["dfk_in", "ifk_in"]
    param_names = ["y", "z"]
    output_names = ["dfk_out", "ifk_out"]

    @staticmethod
    def forward(
        dfk: HomMatrix3D,
        ifk: HomMatrix3D,
        y: Float[torch.Tensor, ""],
        z: Float[torch.Tensor, ""],
    ) -> tuple[HomMatrix3D, HomMatrix3D]:
        hom, hom_inv = hom_rotate_3d(y, z)
        print(hom.dtype, hom_inv.dtype, y.dtype, z.dtype)
        return kinematic_chain_append(dfk, ifk, hom, hom_inv)

    @staticmethod
    def example_inputs(
        dtype: torch.dtype, device: torch.device
    ) -> tuple[HomMatrix3D, HomMatrix3D]:
        return hom_identity_3d(dtype=dtype, device=device)

    @staticmethod
    def example_params(
        dtype: torch.dtype, device: torch.device
    ) -> tuple[
        Float[torch.Tensor, ""],
        Float[torch.Tensor, ""],
    ]:
        return (
            torch.tensor(0.1, dtype=dtype, device=device),
            torch.tensor(0.2, dtype=dtype, device=device),
        )
