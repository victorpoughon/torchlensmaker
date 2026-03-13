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

from typing import Self

import torch
import torch.nn as nn
from jaxtyping import Float

from torchlensmaker.core.base_module import BaseModule
from torchlensmaker.core.tensor_manip import init_param

from .material_kernels import (
    CauchyMaterialKernel,
    NonDispersiveMaterialKernel,
    SellmeierMaterialKernel,
)

# TODO add:
# LinearSegmentedMaterial


class MaterialModel(BaseModule):
    def forward(
        self, wavelength: Float[torch.Tensor, " N"]
    ) -> Float[torch.Tensor, " N"]:
        raise NotImplementedError


class NonDispersiveMaterial(MaterialModel):
    def __init__(
        self,
        n: Float[torch.Tensor, ""] | float,
        trainable: bool = False,
    ):
        super().__init__()
        self.n = init_param(self, "n", n, trainable)
        self.kernel = NonDispersiveMaterialKernel()

    def clone(self, **overrides) -> Self:
        kwargs: dict[str, Any] = dict(n=self.n)
        return type(self)(**kwargs | overrides)

    def __repr__(self) -> str:
        return f"{self._get_name()}(n={self.n})"

    def forward(
        self, wavelength: Float[torch.Tensor, " N"]
    ) -> Float[torch.Tensor, " N"]:
        return self.kernel.apply(wavelength, self.n)


class CauchyMaterial(MaterialModel):
    def __init__(
        self,
        A: Float[torch.Tensor, ""] | float,
        B: Float[torch.Tensor, ""] | float,
        C: Float[torch.Tensor, ""] | float = 0.0,
        D: Float[torch.Tensor, ""] | float = 0.0,
        trainable: bool = False,
    ):
        super().__init__()
        self.A = init_param(self, "A", A, trainable)
        self.B = init_param(self, "B", B, trainable)
        self.C = init_param(self, "C", C, trainable)
        self.D = init_param(self, "D", D, trainable)
        self.kernel = CauchyMaterialKernel()

    def clone(self, **overrides) -> Self:
        kwargs: dict[str, Any] = dict(
            A=self.A,
            B=self.B,
            C=self.C,
            D=self.D,
        )
        return type(self)(**kwargs | overrides)

    def forward(
        self, wavelength: Float[torch.Tensor, " N"]
    ) -> Float[torch.Tensor, " N"]:
        return self.kernel.apply(wavelength, self.A, self.B, self.C, self.D)


class SellmeierMaterial(MaterialModel):
    def __init__(
        self,
        B1: Float[torch.Tensor, ""] | float,
        B2: Float[torch.Tensor, ""] | float,
        B3: Float[torch.Tensor, ""] | float,
        C1: Float[torch.Tensor, ""] | float,
        C2: Float[torch.Tensor, ""] | float,
        C3: Float[torch.Tensor, ""] | float,
        trainable: bool = False,
    ):
        super().__init__()
        self.B1 = init_param(self, "B1", B1, trainable)
        self.B2 = init_param(self, "B2", B2, trainable)
        self.B3 = init_param(self, "B3", B3, trainable)
        self.C1 = init_param(self, "C1", C1, trainable)
        self.C2 = init_param(self, "C2", C2, trainable)
        self.C3 = init_param(self, "C3", C3, trainable)
        self.kernel = SellmeierMaterialKernel()

    def clone(self, **overrides) -> Self:
        kwargs: dict[str, Any] = dict(
            B1=self.B1,
            B2=self.B2,
            B3=self.B3,
            C1=self.C1,
            C2=self.C2,
            C3=self.C3,
        )
        return type(self)(**kwargs | overrides)

    def forward(
        self, wavelength: Float[torch.Tensor, " N"]
    ) -> Float[torch.Tensor, " N"]:
        return self.kernel.apply(
            wavelength, self.B1, self.B2, self.B3, self.C1, self.C2, self.C3
        )
