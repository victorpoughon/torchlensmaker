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

import torch
import torch.nn as nn
import torchlensmaker as tlm

from torchlensmaker.kinematics.homogeneous_geometry import (
    HomMatrix,
    transform_points,
    kinematic_chain_append,
)
from torchlensmaker.kinematics.kinematics_elements import KinematicSequential
from torchlensmaker.materials.get_material_model import (
    MaterialModel,
    get_material_model,
)
from torchlensmaker.elements.sequential import SequentialElement

from typing import Any, Optional, Literal


Tensor = torch.Tensor
Anchor = Literal["origin", "extent"]
Anchors = tuple[Anchor, Anchor]


def lens_thickness_parametrization(
    inner_thickness: Optional[Any], outer_thickness: Optional[Any]
) -> tuple[Any, tuple[Anchor, Anchor]]:
    "Thickness and anchors for the provied thickness parametrization"

    anchors: Anchors
    if inner_thickness is not None and outer_thickness is None:
        thickness = inner_thickness
        anchors = ("origin", "origin")
    elif outer_thickness is not None and inner_thickness is None:
        thickness = outer_thickness
        anchors = ("origin", "extent")
    else:
        raise ValueError("Exactly one of inner/outer thickness must be given")

    return thickness, anchors


class LensMaterialsMixin:
    def __init__(
        self,
        material: str | MaterialModel,
        exit_material: str | MaterialModel = "air",
        **kwargs: Any,
    ):
        material = get_material_model(material)
        exit_material = get_material_model(exit_material)
        super().__init__(**kwargs)
        self.material = material
        self.exit_material = exit_material


class LensBase(LensMaterialsMixin, SequentialElement):
    "A base class to share common lens functions"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def inner_thickness(self) -> Tensor:
        "Thickness at the center of the lens"

        # Infer dtype, device from surface1
        dtype, device = self.surface1.collision_surface.surface.dtype, torch.device("cpu")

        root = tlm.hom_identity_2d(dtype, device)

        A1 = KinematicSequential(
            [
                tlm.ExactKinematicElement2D(
                    *self.surface1.collision_surface.surface_transform(2, dtype)
                )
            ]
        )

        A2 = KinematicSequential(
            [
                tlm.ExactKinematicElement2D(
                    *self.surface1.collision_surface.kinematic_transform(2, dtype)
                ),
                self.gap,
                tlm.ExactKinematicElement2D(
                    *self.surface2.collision_surface.surface_transform(2, dtype)
                ),
            ]
        )

        a1, _ = A1(*root)
        a2, _ = A2(*root)

        root_point = torch.zeros((2,), dtype=dtype)
        p1 = tlm.transform_points(a1, root_point)
        p2 = tlm.transform_points(a2, root_point)

        return (p2 - p1)[0]

    def outer_thickness(self) -> Tensor:
        "Thickness at the outer radius of the lens"

        # Infer dtype, device from surface1
        dtype, device = self.surface1.collision_surface.surface.dtype, torch.device("cpu")

        root = tlm.hom_identity_2d(dtype, device)

        A1 = KinematicSequential(
            [
                tlm.ExactKinematicElement2D(
                    *self.surface1.collision_surface.surface_transform(2, dtype)
                ),
                tlm.Translate2D(x=self.surface1.collision_surface.surface.extent_x()),
            ]
        )

        A2 = KinematicSequential(
            [
                tlm.ExactKinematicElement2D(
                    *self.surface1.collision_surface.kinematic_transform(2, dtype)
                ),
                self.gap,
                tlm.ExactKinematicElement2D(
                    *self.surface2.collision_surface.surface_transform(2, dtype)
                ),
                tlm.Translate2D(x=self.surface2.collision_surface.surface.extent_x()),
            ]
        )

        a1, _ = A1(*root)
        a2, _ = A2(*root)

        root_point = torch.zeros((2,), dtype=dtype)
        p1 = tlm.transform_points(a1, root_point)
        p2 = tlm.transform_points(a2, root_point)

        return (p2 - p1)[0]

    def forward(self, inputs: tlm.OpticalData) -> tlm.OpticalData:
        return tlm.Sequential(self.surface1, self.gap, self.surface2)(inputs)


class Lens(LensBase):
    """
    A lens made of two refractive surfaces with different shapes
    """

    surface1: tlm.RefractiveSurface
    surface2: tlm.RefractiveSurface

    def __init__(
        self,
        surface1: tlm.LocalSurface,
        surface2: tlm.LocalSurface,
        inner_thickness: Optional[float] = None,
        outer_thickness: Optional[float] = None,
        scale1: float = 1.0,
        scale2: float = 1.0,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        thickness, anchors = lens_thickness_parametrization(
            inner_thickness, outer_thickness
        )

        self.surface1 = tlm.RefractiveSurface(
            surface1,
            self.material,
            scale=scale1,
            anchors=anchors,
        )
        self.gap = tlm.Gap(thickness)
        self.surface2 = tlm.RefractiveSurface(
            surface2,
            self.exit_material,
            scale=scale2,
            anchors=(anchors[1], anchors[0]),
        )


class BiLens(LensBase):
    """
    A lens made of two mirrored symmetrical refractive surfaces
    """

    def __init__(
        self,
        surface: tlm.LocalSurface,
        inner_thickness: Optional[float] = None,
        outer_thickness: Optional[float] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        thickness, anchors = lens_thickness_parametrization(
            inner_thickness, outer_thickness
        )

        self.surface1 = tlm.RefractiveSurface(
            surface,
            material=self.material,
            anchors=anchors,
        )
        self.gap = tlm.Gap(thickness)
        self.surface2 = tlm.RefractiveSurface(
            surface,
            material=self.exit_material,
            scale=-1.0,
            anchors=(anchors[1], anchors[0]),
        )


class PlanoLens(Lens):
    """
    A plano-convex or plano-concave lens where one surface is curved
    as the given surface and the other surface is flat.

    By default the first surface is flat and the second is curved.
    This can be switched with the reverse argument:
    * reverse = False (default):  The curved side is the second surface
    * reverse = True:             The curved side is the first surface

    If reverse is true, the surface is flipped.
    """

    def __init__(
        self,
        surface: tlm.LocalSurface,
        inner_thickness: Optional[float] = None,
        outer_thickness: Optional[float] = None,
        reverse: bool = False,
        **kwargs: Any,
    ):
        # TODO use bbox instead of diameter
        plane = tlm.CircularPlane(2 * surface.diameter / 2)
        s1, s2 = (surface, plane) if reverse else (plane, surface)
        super().__init__(
            s1,
            s2,
            inner_thickness,
            outer_thickness,
            scale1=-1.0 if reverse else 1.0,
            **kwargs,
        )
