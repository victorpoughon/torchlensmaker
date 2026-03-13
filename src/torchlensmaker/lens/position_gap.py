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


from dataclasses import dataclass
from typing import Literal, TypeAlias


@dataclass
class InnerGap:
    "Marker for lens gap initialization"

    gap: float


@dataclass
class OuterGap:
    "Marker for lens gap initialization"

    gap: float


PositionGap: TypeAlias = InnerGap | OuterGap


def position_gap_to_anchors(pgap: PositionGap) -> tuple[float, float]:
    "Convert a position gap to a pair of anchors"
    match pgap:
        case InnerGap(_):
            return (0, 0)
        case OuterGap(_):
            return (1, 1)
