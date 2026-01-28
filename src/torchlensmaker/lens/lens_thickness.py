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


from typing import TYPE_CHECKING
from itertools import islice
from jaxtyping import Float
import torch
import torch.nn as nn


from torchlensmaker.kinematics.homogeneous_geometry import (
    hom_identity_2d,
    transform_points,
)
from torchlensmaker.kinematics.kinematics_elements import (
    KinematicSequential,
    ExactKinematicElement2D,
    KinematicElement,
    Translate2D,
)
from torchlensmaker.elements.optical_surfaces import RefractiveSurface

if TYPE_CHECKING:
    from .lens import Lens


def tokinematic(mod: nn.Module) -> KinematicElement:
    if isinstance(mod, KinematicElement):
        return mod
    elif isinstance(mod, RefractiveSurface):
        dtype = mod.collision_surface.surface.dtype
        return ExactKinematicElement2D(
            *mod.collision_surface.kinematic_transform(2, dtype)
        )
    else:
        raise RuntimeError("inner_thickness() got invalid lens")


def lens_inner_thickness(lens: 'Lens') -> Float[torch.Tensor, ""]:
    "Thickness of a lens at the center"

    first_surface, last_surface = lens.sequence[0], lens.sequence[-1]

    # Infer dtype, device from first surface
    # TODO gpu support
    dtype, device = first_surface.collision_surface.surface.dtype, torch.device("cpu")

    # Compute front surface vertex kinematic model
    A1 = KinematicSequential(
        ExactKinematicElement2D(
            *first_surface.collision_surface.surface_transform(2, dtype)
        )
    )

    # Compute rear surface vertex kinematic model
    # by converting the lens sequence to purely kinematic elements
    # except for the last surface
    A2 = KinematicSequential(
        *[tokinematic(mod) for mod in islice(lens.sequence, len(lens.sequence) - 1)],
        ExactKinematicElement2D(
            *last_surface.collision_surface.surface_transform(2, dtype)
        ),
    )

    root = hom_identity_2d(dtype, device)
    a1, _ = A1(*root)
    a2, _ = A2(*root)

    root_point = torch.zeros((2,), dtype=dtype)
    p1 = transform_points(a1, root_point)
    p2 = transform_points(a2, root_point)

    return (p2 - p1)[0]


def lens_outer_thickness(lens: 'Lens') -> Float[torch.Tensor, ""]:
    "Thickness of a lens at the edge"

    front_surface, rear_surface = lens.sequence[0], lens.sequence[-1]

    # Infer dtype, device from first surface
    # TODO gpu support
    dtype, device = front_surface.collision_surface.surface.dtype, torch.device("cpu")

    # Compute front surface vertex kinematic model
    A1 = KinematicSequential(
        ExactKinematicElement2D(
            *front_surface.collision_surface.surface_transform(2, dtype)
        ),
        Translate2D(x=front_surface.collision_surface.surface.extent_x()),
    )

    # Compute rear surface vertex kinematic model
    # by converting the lens sequence to purely kinematic elements
    # except for the last surface
    A2 = KinematicSequential(
        *[tokinematic(mod) for mod in islice(lens.sequence, len(lens.sequence) - 1)],
        ExactKinematicElement2D(
            *rear_surface.collision_surface.surface_transform(2, dtype)
        ),
        Translate2D(x=rear_surface.collision_surface.surface.extent_x()),
    )

    root = hom_identity_2d(dtype, device)
    a1, _ = A1(*root)
    a2, _ = A2(*root)

    root_point = torch.zeros((2,), dtype=dtype)
    p1 = transform_points(a1, root_point)
    p2 = transform_points(a2, root_point)

    return (p2 - p1)[0]
