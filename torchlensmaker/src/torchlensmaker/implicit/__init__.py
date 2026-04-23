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

from .implicit_circle import (
    implicit_yzcircle_2d,
    implicit_yzcircle_3d,
)
from .implicit_disk import (
    implicit_disk_2d,
    implicit_disk_3d,
)
from .implicit_plane import (
    implicit_yaxis_2d,
    implicit_yzplane_3d,
)
from .types import ImplicitFunction, ImplicitResult

__all__ = [
    "implicit_disk_2d",
    "implicit_disk_3d",
    "implicit_yaxis_2d",
    "implicit_yzcircle_2d",
    "implicit_yzcircle_3d",
    "implicit_yzplane_3d",
    "ImplicitFunction",
    "ImplicitResult",
]
