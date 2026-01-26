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

from enum import Enum


class Dim(Enum):
    """
    Enum to represent the physical dimensionality of a model
    can be:
        * TWO: the model represents two physical dimensions
        * THREE: the model represents three physical dimensions
        * MIXED: the model represents two or three physical dimensions
    """

    ONE = 1
    TWO = 2
    THREE = 3
    MIXED = 4
