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


from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from torchlensmaker.core.ray_bundle import RayBundle
from torchlensmaker.types import Tf


@dataclass
class OpticalScene:
    rays: OrderedDict[str, RayBundle]
    chain: OrderedDict[str, Tf]
    surfaces: OrderedDict[str, tuple[Tf, Any]]
