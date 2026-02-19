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


from jaxtyping import Float
import torch

from torchlensmaker.core.functional_kernel import FunctionalKernel
from torchlensmaker.types import ScalarTensor, BatchTensor


class NonDispersiveMaterialKernel(FunctionalKernel):
    "Material model for a constant index of refraction"

    inputs = {"wavelength": BatchTensor}
    params = {"n": ScalarTensor}
    outputs = {"index": BatchTensor}

    @staticmethod
    def apply(
        wavelength: Float[torch.Tensor, " N"],
        n: Float[torch.Tensor, ""],
    ) -> Float[torch.Tensor, " N"]:
        return n.expand_as(wavelength)

    @staticmethod
    def example_inputs(
        dtype: torch.dtype, device: torch.device
    ) -> tuple[torch.Tensor, ...]:
        return (torch.tensor([400, 401, 402, 403], dtype=dtype, device=device),)

    @staticmethod
    def example_params(
        dtype: torch.dtype, device: torch.device
    ) -> tuple[torch.Tensor, ...]:
        return (torch.tensor(1.5, dtype=dtype),)


class CauchyMaterialKernel(FunctionalKernel):
    "Material model using Cauchy's equation with four coefficents"

    inputs = {"wavelength": BatchTensor}
    params = {
        "A": ScalarTensor,
        "B": ScalarTensor,
        "C": ScalarTensor,
        "D": ScalarTensor,
    }
    outputs = {"index": BatchTensor}

    @staticmethod
    def apply(
        wavelength: Float[torch.Tensor, " N"],
        A: Float[torch.Tensor, ""],
        B: Float[torch.Tensor, ""],
        C: Float[torch.Tensor, ""],
        D: Float[torch.Tensor, ""],
    ) -> Float[torch.Tensor, " N"]:
        wavelength_micro = wavelength / 1000
        index: Float[torch.Tensor, " N"] = (
            A.expand_as(wavelength)
            + B / torch.pow(wavelength_micro, 2)
            + C / torch.pow(wavelength_micro, 4)
            + D / torch.pow(wavelength_micro, 6)
        )
        return index

    @staticmethod
    def example_inputs(
        dtype: torch.dtype, device: torch.device
    ) -> tuple[torch.Tensor, ...]:
        return (torch.tensor([400, 401, 402, 403], dtype=dtype, device=device),)

    @staticmethod
    def example_params(
        dtype: torch.dtype, device: torch.device
    ) -> tuple[torch.Tensor, ...]:
        return (
            torch.tensor(1.31044, dtype=dtype),
            torch.tensor(0.0050572226, dtype=dtype),
            torch.tensor(0.0, dtype=dtype),
            torch.tensor(0.0, dtype=dtype),
        )


class SellmeierMaterialKernel(FunctionalKernel):
    "Material model using Sellmeier equation with six coefficents"

    inputs = {"wavelength": BatchTensor}
    params = params = {
        "B1": ScalarTensor,
        "B2": ScalarTensor,
        "B3": ScalarTensor,
        "C1": ScalarTensor,
        "C2": ScalarTensor,
        "C3": ScalarTensor,
    }
    outputs = {"index": BatchTensor}

    @staticmethod
    def apply(
        wavelength: Float[torch.Tensor, " N"],
        B1: Float[torch.Tensor, ""],
        B2: Float[torch.Tensor, ""],
        B3: Float[torch.Tensor, ""],
        C1: Float[torch.Tensor, ""],
        C2: Float[torch.Tensor, ""],
        C3: Float[torch.Tensor, ""],
    ) -> Float[torch.Tensor, " N"]:
        W = wavelength / 1000
        W2 = torch.pow(W, 2)
        index: Float[torch.Tensor, " N"] = torch.sqrt(
            1 + (B1 * W2) / (W2 - C1) + (B2 * W2) / (W2 - C2) + (B3 * W2) / (W2 - C3)
        )
        return index

    @staticmethod
    def example_inputs(
        dtype: torch.dtype, device: torch.device
    ) -> tuple[torch.Tensor, ...]:
        return (torch.tensor([400, 401, 402, 403], dtype=dtype, device=device),)

    @staticmethod
    def example_params(
        dtype: torch.dtype, device: torch.device
    ) -> tuple[torch.Tensor, ...]:
        return (
            torch.tensor(0.6961663, dtype=dtype),
            torch.tensor(0.4079426, dtype=dtype),
            torch.tensor(0.8974794, dtype=dtype),
            torch.tensor(0.00467914825, dtype=dtype),
            torch.tensor(0.01351206307, dtype=dtype),
            torch.tensor(97.9340025379, dtype=dtype),
        )
