# This file is part of Torch Lens Maker
# Copyright (C) 2025 Victor Poughon
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
from copy import copy

Tensor = torch.Tensor

default_material_model_name = "(unnamed)"


class MaterialModel:
    def __init__(self, name: str = default_material_model_name):
        self.name = name

    def is_dispersive(self) -> bool:
        raise NotImplementedError

    def refractive_index(self, W: Tensor) -> Tensor:
        "Refractive index function of wavelength (in nm)"
        raise NotImplementedError


class NonDispersiveMaterial(MaterialModel):
    def __init__(self, n: float, name: str = default_material_model_name):
        super().__init__(name)
        self.n = n

    def is_dispersive(self):
        return False

    def refractive_index(self, W: Tensor) -> Tensor:
        return torch.full_like(W, self.n)

    def __str__(self) -> str:
        return f"{type(self).__name__}(name={repr(self.name)}, n={self.n})"


class CauchyMaterial(MaterialModel):
    def __init__(
        self,
        A: float,
        B: float,
        C: float = 0,
        D: float = 0,
        name: str = default_material_model_name,
    ):
        super().__init__(name)
        self.A = A
        self.B = B
        self.C = C
        self.D = D

    def is_dispersive(self):
        return True

    def refractive_index(self, W: Tensor) -> Tensor:
        Wmicro = W / 1000
        return (
            torch.full_like(Wmicro, self.A)
            + self.B / torch.pow(Wmicro, 2)
            + self.C / torch.pow(Wmicro, 4)
            + self.D / torch.pow(Wmicro, 6)
        )

    def __str__(self) -> str:
        return f"{type(self).__name__}(name={repr(self.name)}, A={self.A}, B={self.B}, C={self.C}, D={self.D})"


class SellmeierMaterial(MaterialModel):
    def __init__(
        self,
        B1: float,
        B2: float,
        B3: float,
        C1: float,
        C2: float,
        C3: float,
        name: str = default_material_model_name,
    ):
        super().__init__(name)
        self.B1, self.B2, self.B3 = B1, B2, B3
        self.C1, self.C2, self.C3 = C1, C2, C3

    def is_dispersive(self):
        return True

    def refractive_index(self, W) -> Tensor:
        # wavelength in micrometers
        Wm = W / 1000
        W2 = torch.pow(Wm, 2)

        return torch.sqrt(
            1
            + (self.B1 * W2) / (W2 - self.C1)
            + (self.B2 * W2) / (W2 - self.C2)
            + (self.B3 * W2) / (W2 - self.C3)
        )

    def __str__(self) -> str:
        return f"{type(self).__name__}(name={repr(self.name)}, B1={self.B1}, B2={self.B2}, B3={self.B3}, C1={self.C1}, C2={self.C2}, C3={self.C3}"


# TODO add:
# LinearSegmentedMaterial


# These are "default" materials in the sense that they are the models
# that will be picked when using a string material argument, as in: material="water"
# any material argument can also be a MaterialModel object directly for more advanced usage
default_material_models = [
    NonDispersiveMaterial(1.0, name="vacuum"),
    NonDispersiveMaterial(1.00027, name="air"),
    # Bashkatov, Alexey & Genina, Elina. (2003).
    # Water refractive index in dependence on temperature and wavelength:
    # A simple approximation. Proceedings of SPIE
    # The International Society for Optical Engineering. 5068. 10.1117/12.518857.
    CauchyMaterial(1.31984, 0.005190553, name="water20C"),
    CauchyMaterial(
        1.31044,
        0.0050572226,
        name="water80C",
    ),
    CauchyMaterial(
        1.31984,
        0.005190553,
        name="water",
    ),
    CauchyMaterial(1.31044, 0.0050572226, name="Tea Earl Grey Hot"),
    # https://en.wikipedia.org/wiki/Cauchy%27s_equation
    CauchyMaterial(1.4580, 0.00354, name="Fused Silica"),
    CauchyMaterial(1.5046, 0.00420, name="BK7"),
    CauchyMaterial(1.5220, 0.00459, name="K5"),
    CauchyMaterial(1.5690, 0.00531, name="BaK4"),
    CauchyMaterial(1.6700, 0.00743, name="BaF10"),
    CauchyMaterial(1.7280, 0.013420, name="SF10"),
]

# Generate -nd suffix versions for "non dispersive" alternatives
# by taking the refractive index at 500nm
for model in copy(default_material_models):
    if model.is_dispersive():
        name = model.name + "-nd"
        n = model.refractive_index(torch.tensor(500))
        default_material_models.append(NonDispersiveMaterial(n, name=name))

default_material_models_dict = {
    model.name: model
    for model in default_material_models
    if model.name != default_material_model_name
}


def get_material_model(material: str | MaterialModel) -> MaterialModel:
    if isinstance(material, MaterialModel):
        return material

    try:
        return default_material_models_dict[material]
    except KeyError:
        raise ValueError(f"No material model for '{material}'")
