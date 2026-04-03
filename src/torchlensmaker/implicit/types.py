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

from dataclasses import dataclass, field
from typing import Callable, TypeAlias

import torch


@dataclass
class ImplicitResult:
    val: torch.Tensor
    grad: torch.Tensor | None = field(default=None)
    hess: torch.Tensor | None = field(default=None)


ImplicitFunction: TypeAlias = Callable[[torch.Tensor], ImplicitResult]
