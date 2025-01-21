import torch
import torch.nn as nn
import torchlensmaker as tlm

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

    dim, dtype = transform.dim, transform.dtype

    # Get surface local point corresponding to anchor
    if anchor == "origin":
        point = surface.zero(dim, dtype)
    elif anchor == "extent":
        point = surface.extent(dim, dtype)

    # Transform it to absolute space
    return transform.direct_points(point)


def anchor_thickness(
    lens: nn.Sequential, anchor: Anchor, dim: int, dtype: torch.dtype
) -> Tensor:
    "Thickness of a lens at an anchor"

    # Evaluate the lens stack with zero rays, just to compute the transforms
    # TODO make rays variable dimensions not needed here
    execute_list, _ = tlm.full_forward(
        lens, tlm.default_input({"dim": dim, "dtype": dtype, "base": 0})
    )

    s1_transform = tlm.forward_kinematic(
        execute_list[0].inputs.transforms + lens[0].surface_transform(dim, dtype)
    )
    s2_transform = tlm.forward_kinematic(
        execute_list[2].inputs.transforms + lens[2].surface_transform(dim, dtype)
    )

    a1 = anchor_abs(lens[0].surface, s1_transform, anchor)
    a2 = anchor_abs(lens[2].surface, s2_transform, anchor)

    return torch.linalg.vector_norm(a1 - a2)  # type: ignore


class LensBase(nn.Module):
    "A base class to share common lens functions"

    optics: nn.Sequential

    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs: tlm.OpticalData) -> tlm.OpticalData:
        return self.optics(inputs)  # type: ignore

    def inner_thickness(self) -> Tensor:
        "Thickness at the center of the lens"
        return anchor_thickness(self.optics, "origin", 3, torch.float64)

    def outer_thickness(self) -> Tensor:
        "Thickness at the outer radius of the lens"
        return anchor_thickness(self.optics, "extent", 3, torch.float64)


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
        n: tuple[float, float],
        inner_thickness: Optional[float] = None,
        outer_thickness: Optional[float] = None,
    ):
        super().__init__()

        thickness, anchors = lens_thickness_parametrization(
            inner_thickness, outer_thickness
        )

        self.surface1 = tlm.RefractiveSurface(surface1, n, anchors=anchors)
        self.gap = tlm.Gap(thickness)
        self.surface2 = tlm.RefractiveSurface(
            surface2,
            (n[1], n[0]),
            anchors=(anchors[1], anchors[0]),
        )

        self.optics = nn.Sequential(self.surface1, self.gap, self.surface2)


class BiLens(LensBase):
    """
    A lens made of two mirrored symmetrical refractive surfaces
    """

    def __init__(
        self,
        surface: tlm.LocalSurface,
        n: tuple[float, float],
        inner_thickness: Optional[float] = None,
        outer_thickness: Optional[float] = None,
    ):
        super().__init__()

        thickness, anchors = lens_thickness_parametrization(
            inner_thickness, outer_thickness
        )

        self.surface1 = tlm.RefractiveSurface(
            surface,
            n,
            anchors=anchors,
        )
        self.gap = tlm.Gap(thickness)
        self.surface2 = tlm.RefractiveSurface(
            surface,
            (n[1], n[0]),
            scale=-1.0,
            anchors=(anchors[1], anchors[0]),
        )

        self.optics = nn.Sequential(self.surface1, self.gap, self.surface2)


class PlanoLens(Lens):
    """
    A plano-convex or plano-concave lens where one surface is curved
    as the given surface and the other surface is flat.

    By default the first surface is flat and the second is curved.
    This can be switched with the reverse argument:
    * reverse = False (default):  The curved side is the second surface
    * reverse = True:             The curved side is the first surface
    """

    def __init__(
        self,
        surface: tlm.LocalSurface,
        n: tuple[float, float],
        inner_thickness: Optional[float] = None,
        outer_thickness: Optional[float] = None,
        reverse: bool = False,
    ):
        plane = tlm.CircularPlane(2 * surface.outline.max_radius())
        s1, s2 = (surface, plane) if reverse else (plane, surface)
        super().__init__(s1, s2, n, inner_thickness, outer_thickness)
