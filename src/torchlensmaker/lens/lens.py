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

from typing import Any, Self, cast

import torch
import torch.nn as nn
from jaxtyping import Float

from torchlensmaker.core.base_module import BaseModule
from torchlensmaker.sequential.sequential import Sequential
from torchlensmaker.sequential.sequential_data import SequentialData
from torchlensmaker.sequential.sequential_element import SequentialElement
from torchlensmaker.optical_surfaces.refractive_surface import RefractiveSurface

from .lens_thickness import (
    lens_inner_thickness,
    lens_minimal_diameter,
    lens_outer_thickness,
)


class Lens(Sequential):
    @property
    def dtype(self) -> torch.dtype:
        return cast(torch.dtype, self[1].x.dtype)

    @property
    def device(self) -> torch.device:
        return cast(torch.device, self[1].x.device)

    @property
    def first_surface(self) -> RefractiveSurface:
        return cast(RefractiveSurface, self[0])

    @property
    def last_surface(self) -> RefractiveSurface:
        return cast(RefractiveSurface, self[-1])

    def inner_thickness(self) -> Float[torch.Tensor, ""]:
        return lens_inner_thickness(self)

    def outer_thickness(self) -> Float[torch.Tensor, ""]:
        return lens_outer_thickness(self)

    def minimal_diameter(self) -> Float[torch.Tensor, ""]:
        return lens_minimal_diameter(self)
