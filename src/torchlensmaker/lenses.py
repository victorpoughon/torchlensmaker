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
import torch.nn as nn
import torchlensmaker as tlm

from torchlensmaker.materials import MaterialModel, get_material_model

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


def anchor_abs(
    surface: tlm.LocalSurface, transform: tlm.TransformBase, anchor: Anchor
) -> Tensor:
    "Get absolute position of a surface anchor"

    dim = transform.dim
    assert surface.dtype == transform.dtype

    # Get surface local point corresponding to anchor
    if anchor == "origin":
        point = surface.zero(dim)
    elif anchor == "extent":
        point = surface.extent(dim)

    # Transform it to absolute space
    return transform.direct_points(point)


def anchor_thickness(
    lens: nn.Module, anchor: Anchor, dim: int, dtype: torch.dtype
) -> Tensor:
    "Thickness of a lens at an anchor"

    # Evaluate the lens stack with zero rays, just to compute the transforms
    input_tree, output_tree = tlm.forward_tree(
        lens, tlm.default_input(sampling={}, dim=dim, dtype=dtype)
    )

    s1_transform = tlm.forward_kinematic(
        input_tree[lens.surface1].transforms
        + lens.surface1.collision_surface.surface_transform(dim, dtype)
    )
    s2_transform = tlm.forward_kinematic(
        input_tree[lens.surface2].transforms
        + lens.surface2.collision_surface.surface_transform(dim, dtype)
    )

    a1 = anchor_abs(lens.surface1.surface, s1_transform, anchor)
    a2 = anchor_abs(lens.surface2.surface, s2_transform, anchor)

    return torch.linalg.vector_norm(a1 - a2)  # type: ignore


class LensMaterialsMixin:
    def __init__(
        self,
        material: str | MaterialModel,
        exit_material: str | MaterialModel = "air",
        **kwargs: Any,
    ):
        self.material = get_material_model(material)
        self.exit_material = get_material_model(exit_material)
        super().__init__(**kwargs)


class LensBase(LensMaterialsMixin, nn.Module):
    "A base class to share common lens functions"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def inner_thickness(self) -> Tensor:
        "Thickness at the center of the lens"
        return anchor_thickness(self, "origin", 3, torch.float64)

    def outer_thickness(self) -> Tensor:
        "Thickness at the outer radius of the lens"
        return anchor_thickness(self, "extent", 3, torch.float64)

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
