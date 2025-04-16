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

from torchlensmaker.surfaces.sphere_r import LocalSurface

from torchlensmaker.core.transforms import (
    TransformBase,
    LinearTransform,
    TranslateTransform,
    ComposeTransform,
)

from typing import Callable

from torchlensmaker.core.rot3d import euler_angles_to_matrix
from torchlensmaker.core.rot2d import rotation_matrix_2D


def basic_transform(
    scale: float,
    anchor: str,
    thetas: float | list[float],
    translate: list[float],
    dtype: torch.dtype = torch.float64,
) -> Callable[[LocalSurface], TransformBase]:
    """
    Compound transform used for testing

    Transform is of the form: Y = RS(X - A) + T

    Returns a function foo(surface)
    """

    if isinstance(thetas, list) and len(translate) == 3:
        dim = 3
    elif isinstance(thetas, (float, int)) and len(translate) == 2:
        dim = 2
    else:
        raise RuntimeError("invalid arguments to basic_transform")

    def makeit(surface: LocalSurface) -> TransformBase:
        # anchor
        assert dtype == surface.dtype
        anchor_translate = surface.extent(dim)
        transforms: list[TransformBase] = (
            [TranslateTransform(-anchor_translate)] if anchor == "extent" else []
        )

        # scale
        Md = torch.eye(dim, dtype=dtype) * scale
        Mi = torch.eye(dim, dtype=dtype) * 1 / scale
        transforms.append(LinearTransform(Md, Mi))

        # rotate
        if dim == 2:
            Mr = rotation_matrix_2D(torch.deg2rad(torch.as_tensor(thetas, dtype=dtype)))
        else:
            Mr = euler_angles_to_matrix(
                torch.deg2rad(torch.as_tensor(thetas, dtype=dtype)), "XYZ"
            ).to(dtype=dtype)  # TODO need to support dtype in euler_angles_to_matrix

        transforms.append(LinearTransform(Mr, Mr.T))

        # translate
        transforms.append(TranslateTransform(torch.as_tensor(translate, dtype=dtype)))

        return ComposeTransform(transforms)

    return makeit
