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

from copy import copy
import torch

from .material_elements import MaterialModel, NonDispersiveMaterial, CauchyMaterial


# These are "default" materials in the sense that they are the models
# that will be picked when using a string material argument, as in: material="water"
# any material argument can also be a MaterialModel object directly for more advanced usage
default_material_models: dict[str, MaterialModel] = {
    "vacuum": NonDispersiveMaterial(1.0),
    "air": NonDispersiveMaterial(1.00027),
    "water": CauchyMaterial(
        1.31984,
        0.005190553,
    ),
    "water20C": CauchyMaterial(1.31984, 0.005190553),
    "water80C": CauchyMaterial(
        1.31044,
        0.0050572226,
    ),
    "Tea Earl Grey Hot": CauchyMaterial(1.31044, 0.0050572226),
    # https://en.wikipedia.org/wiki/Cauchy%27s_equation
    "Fused Silica": CauchyMaterial(1.4580, 0.00354),
    "BK7": CauchyMaterial(1.5046, 0.00420),
    "K5": CauchyMaterial(1.5220, 0.00459),
    "BaK4": CauchyMaterial(1.5690, 0.00531),
    "BaF10": CauchyMaterial(1.6700, 0.00743),
    "SF10": CauchyMaterial(1.7280, 0.013420),
}


def get_material_model(material: str | MaterialModel) -> MaterialModel:
    if isinstance(material, MaterialModel):
        return material

    try:
        return default_material_models[material]
    except KeyError:
        raise ValueError(f"No material model for '{material}'")
