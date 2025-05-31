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

from typing import Any, Callable


from torchlensmaker.core.transforms import forward_kinematic, TransformBase


from torchlensmaker.analysis.colors import (
    color_valid,
    color_focal_point,
    color_blocked,
)

from . import tlmviewer

from .rendering import Collective, ray_variables_dict
from .rendering import Artist

Tensor = torch.Tensor


LAYER_VALID_RAYS = 1
LAYER_BLOCKED_RAYS = 2
LAYER_OUTPUT_RAYS = 3
LAYER_JOINTS = 4


def render_rays_until(
    P: Tensor,
    V: Tensor,
    end: Tensor,
    variables: dict[str, Tensor],
    domain: dict[str, list[float]],
    default_color: str,
    layer: int,
) -> list[Any]:
    "Render rays until an absolute X coordinate"
    assert end.dim() == 0
    # div by zero here for vertical rays
    t = (end - P[:, 0]) / V[:, 0]
    ends = P + t.unsqueeze(1).expand_as(V) * V
    return [
        tlmviewer.render_rays(
            P,
            ends,
            variables=variables,
            domain=domain,
            default_color=default_color,
            layer=layer,
        )
    ]


def render_rays_length(
    P: Tensor,
    V: Tensor,
    length: float | Tensor,
    variables: dict[str, Tensor],
    domain: dict[str, list[float]],
    layer: int,
    default_color: str = color_valid,
) -> list[Any]:
    "Render rays with fixed length"

    if isinstance(length, Tensor):
        assert length.dim() in {0, 1}

    if isinstance(length, Tensor) and length.dim() == 1:
        length = length.unsqueeze(1).expand_as(V)

    return [
        tlmviewer.render_rays(
            P,
            P + length * V,
            variables=variables,
            domain=domain,
            default_color=default_color,
            layer=layer,
        )
    ]


class ForwardArtist(Artist):
    "Forward rendering to a subobject"

    def __init__(self, getter: Callable[[nn.Module], nn.Module]):
        super().__init__()
        self.getter = getter

    def render_module(self, collective: "Collective", module: nn.Module) -> list[Any]:
        return collective.render_module(self.getter(module))

    def render_rays(self, collective: "Collective", module: nn.Module) -> list[Any]:
        return collective.render_rays(self.getter(module))


class CollisionSurfaceArtist(Artist):
    def render_module(self, collective: "Collective", module: nn.Module) -> list[Any]:
        inputs = collective.input_tree[module]
        dim, dtype = inputs.dim, inputs.dtype

        chain = inputs.transforms + module.surface_transform(dim, dtype)
        tf = forward_kinematic(chain)

        return [tlmviewer.render_surface(module.surface, tf, dim)]

    def render_rays(self, collective: Collective, module: nn.Module) -> list[Any]:
        inputs = collective.input_tree[module]
        t: Tensor
        normals: Tensor
        valid: Tensor
        new_chain: list[TransformBase]
        t, normals, valid, new_chain = collective.output_tree[module]

        collision_points = inputs.P + t.unsqueeze(1).expand_as(inputs.V) * inputs.V
        hits = valid.sum()
        misses = (~valid).sum()

        # Render hit rays
        rays_hit = (
            [
                tlmviewer.render_rays(
                    inputs.P[valid],
                    collision_points[valid],
                    variables=ray_variables_dict(
                        inputs, collective.ray_variables.variables, valid
                    ),
                    domain=collective.ray_variables.domain,
                    default_color=color_valid,
                    layer=LAYER_VALID_RAYS,
                )
            ]
            if hits > 0
            else []
        )

        # Render miss rays - rays absorbed because not colliding
        rays_miss = (
            render_rays_until(
                inputs.P[~valid],
                inputs.V[~valid],
                inputs.target()[0],
                variables=ray_variables_dict(
                    inputs, collective.ray_variables.variables, ~valid
                ),
                domain=collective.ray_variables.domain,
                default_color=color_blocked,
                layer=LAYER_BLOCKED_RAYS,
            )
            if misses > 0
            else []
        )

        return rays_hit + rays_miss


class RefractiveSurfaceArtist(Artist):
    def render_module(self, collective: "Collective", module: nn.Module) -> list[Any]:
        return collective.render_module(module.collision_surface)

    def render_rays(self, collective: Collective, module: nn.Module) -> list[Any]:
        inputs = collective.input_tree[module]
        output, valid_refraction = collective.output_tree[module]

        t, _, collision_valid, _ = collective.output_tree[module.collision_surface]
        collision_points = inputs.P + t.unsqueeze(1).expand_as(inputs.V) * inputs.V
        tir_mask = torch.logical_and(~valid_refraction, collision_valid)

        # render tir absorbed rays
        # TODO make a ray type for it in tlmviewer
        if module._tir == "absorb" and tir_mask.sum() > 0:
            rays_tir = [
                tlmviewer.render_rays(
                    inputs.P[tir_mask],
                    collision_points[tir_mask],
                    variables=ray_variables_dict(
                        inputs, collective.ray_variables.variables, tir_mask
                    ),
                    domain=collective.ray_variables.domain,
                    default_color="pink",
                    layer=LAYER_VALID_RAYS,  # TODO remove layers
                )
            ]
        else:
            rays_tir = []

        return collective.render_rays(module.collision_surface) + rays_tir


class FocalPointArtist(Artist):
    def render_module(self, collective: "Collective", module: nn.Module) -> list[Any]:
        target = collective.input_tree[module].target().unsqueeze(0)
        return [tlmviewer.render_points(target, color_focal_point)]

    def render_rays(self, collective: Collective, module: nn.Module) -> list[Any]:
        inputs = collective.input_tree[module]

        # Distance from ray origin P to target
        dist = torch.linalg.vector_norm(inputs.P - inputs.target(), dim=1)

        # Always draw rays in their positive t direction
        t = torch.abs(dist)
        return render_rays_length(
            inputs.P,
            inputs.V,
            t,
            layer=LAYER_VALID_RAYS,
            variables=ray_variables_dict(inputs, collective.ray_variables.variables),
            domain=collective.ray_variables.domain,
            default_color=color_valid,
        )


class EndArtist(Artist):
    def __init__(self, end: float):
        self.end = end

    def render_rays(self, collective: Collective, module: nn.Module) -> list[Any]:
        return render_rays_length(
            collective.output_tree[module].P,
            collective.output_tree[module].V,
            self.end,
            variables=ray_variables_dict(
                collective.output_tree[module], collective.ray_variables.variables
            ),
            domain=collective.ray_variables.domain,
            default_color=color_valid,
            layer=LAYER_OUTPUT_RAYS,
        )


class SequentialArtist(Artist):
    def render_module(self, collective: "Collective", module: nn.Module) -> list[Any]:
        nodes = []
        for child in module.children():
            nodes.extend(collective.render_module(child))
        return nodes

    def render_rays(self, collective: Collective, module: nn.Module) -> list[Any]:
        nodes = []
        for child in module.children():
            nodes.extend(collective.render_rays(child))
        return nodes


class LensArtist(Artist):
    def render_module(self, collective: "Collective", module: nn.Module) -> list[Any]:
        nodes = []
        nodes.extend(collective.render_module(module.surface1))
        nodes.extend(collective.render_module(module.surface2))
        return nodes

    def render_rays(self, collective: Collective, module: nn.Module) -> list[Any]:
        nodes = []
        nodes.extend(collective.render_rays(module.surface1))
        nodes.extend(collective.render_rays(module.surface2))
        return nodes
