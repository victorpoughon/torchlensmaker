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
        point = surface.extent_point(dim, dtype)

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
