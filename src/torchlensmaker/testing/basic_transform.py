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


from typing import Callable

import torch

from torchlensmaker.surfaces.sphere_r import LocalSurface
from torchlensmaker.new_kinematics.homogeneous_geometry import (
    HomMatrix,
    hom_matrix_3d,
    hom_matrix,
    hom_identity_2d,
    hom_identity_3d,
    hom_translate_2d,
    hom_translate_3d,
    hom_rotate_2d,
    hom_rotate_3d,
    hom_compose,
)
from torchlensmaker.core.rot3d import euler_angles_to_matrix


def basic_transform(
    scale: float,
    anchor: str,
    thetas: float | list[float],
    translate: list[float],
    dtype: torch.dtype = torch.float64,
) -> Callable[[LocalSurface], tuple[HomMatrix, HomMatrix]]:
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

    def makeit2d(surface: LocalSurface) -> tuple[HomMatrix, HomMatrix]:
        dtype = surface.dtype
        transforms: list[tuple[HomMatrix, HomMatrix]] = []

        # anchor
        anchor_translate = surface.extent(dim)

        if anchor == "extent":
            if dim == 2:
                transforms.append(hom_translate_2d(-anchor_translate))
            elif dim == 3:
                transforms.append(hom_translate_3d(-anchor_translate))

        # scale
        Md = hom_matrix(torch.eye(dim, dtype=dtype) * scale)
        Mi = hom_matrix(torch.eye(dim, dtype=dtype) * 1.0 / scale)
        transforms.append((Md, Mi))

        # rotate
        if dim == 2:
            transforms.append(
                hom_rotate_2d(torch.deg2rad(torch.as_tensor(thetas, dtype=dtype)))
            )
        elif dim == 3:
            Mr = euler_angles_to_matrix(
                torch.deg2rad(torch.as_tensor(thetas, dtype=dtype)), "XYZ"
            ).to(dtype=dtype)  # TODO need to support dtype in euler_angles_to_matrix
            Hrot, Hrot_inv = hom_matrix_3d(Mr), hom_matrix_3d(Mr.T)

            transforms.append((Hrot, Hrot_inv))

        # translate
        if dim == 2:
            transforms.append(hom_translate_2d(torch.as_tensor(translate, dtype=dtype)))
        elif dim == 3:
            transforms.append(hom_translate_3d(torch.as_tensor(translate, dtype=dtype)))

        homs = [h for (h, hi) in transforms]
        homs_inv = [hi for (h, hi) in transforms]
        return hom_compose(homs, homs_inv)

    return makeit2d
