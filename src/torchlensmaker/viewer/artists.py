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

from typing import Any, Callable

from torchlensmaker.kinematics.homogeneous_geometry import (
    kinematic_chain_append,
    transform_points,
)

from . import tlmviewer

from .rendering import Collective
from .rendering import Artist

Tensor = torch.Tensor


class ForwardArtist(Artist):
    "Forward rendering to a subobject"

    def __init__(self, getter: Callable[[nn.Module], nn.Module]):
        super().__init__()
        self.getter = getter

    def render(self, collective: "Collective", module: nn.Module) -> list[Any]:
        return collective.render(self.getter(module))


class ReflectiveSurfaceArtist(Artist):
    def render(self, collective: "Collective", module: nn.Module) -> list[Any]:
        # Render module
        inputs = collective.input_tree[module]
        t, normals, valid, fk_surface, fk_next = collective.output_tree[module.surface]

        # Render surface
        rendered_surface = module.surface.render()
        rendered_surface["matrix"] = fk_surface.direct.tolist()

        # Render rays
        rendered_rays = tlmviewer.render_hit_miss_rays(
            inputs.rays.P,
            inputs.rays.V,
            t,
            inputs.target()[0],
            valid,
            variables_hit=inputs.ray_variables_dict(valid),
            variables_miss=inputs.ray_variables_dict(~valid),
            domain=collective.ray_variables_domains,
        )

        # Render joints
        rendered_joints = tlmviewer.render_joint(inputs.fk.direct)

        return [rendered_surface] + rendered_rays + rendered_joints


class RefractiveSurfaceArtist(Artist):
    def render(self, collective: "Collective", module: nn.Module) -> list[Any]:
        # Render module
        inputs = collective.input_tree[module]
        
        ## little problem here, cannot reuse surface because it will mess up collective io tree
        t, normals, collision_valid, fk_surface, fk_next = collective.output_tree[module.surface]

        # Render surface
        rendered_surface = module.surface.render()
        rendered_surface["matrix"] = fk_surface.direct.tolist()

        # Render rays
        rendered_rays = tlmviewer.render_hit_miss_rays(
            inputs.rays.P,
            inputs.rays.V,
            t,
            inputs.target()[0],
            collision_valid,
            variables_hit=inputs.ray_variables_dict(collision_valid),
            variables_miss=inputs.ray_variables_dict(~collision_valid),
            domain=collective.ray_variables_domains,
        )

        # Render joints
        rendered_joints = tlmviewer.render_joint(inputs.fk.direct)

        # TODO reenable
        # Render TIR absorbed rays
        # TODO make a ray type for it in tlmviewer
        # _, valid_refraction = collective.output_tree[module.refractive_interface]
        # tir_mask = torch.logical_and(~valid_refraction, collision_valid)
        # if module._tir_mode == "absorb" and tir_mask.sum() > 0:
        #     collision_points = inputs.rays.points_at(t)
        #     rays_tir = [
        #         tlmviewer.render_rays(
        #             inputs.rays.P[tir_mask],
        #             collision_points[tir_mask],
        #             variables=inputs.ray_variables_dict(tir_mask),
        #             domain=collective.ray_variables_domains,
        #             default_color="pink",
        #             layer=tlmviewer.LAYER_VALID_RAYS,  # TODO remove layers
        #         )
        #     ]
        # else:
        #     rays_tir = []
        rays_tir = []

        return [rendered_surface] + rendered_rays + rendered_joints + rays_tir


class ApertureArtist(Artist):
    def render(self, collective: "Collective", module: nn.Module) -> list[Any]:
        # Render module
        inputs = collective.input_tree[module]
        t, normals, valid, fk_surface, fk_next = collective.output_tree[module.surface]

        # Render surface
        rendered_surface = module.surface.render()
        rendered_surface["matrix"] = fk_surface.direct.tolist()

        # Render rays
        rendered_rays = tlmviewer.render_hit_miss_rays(
            inputs.rays.P,
            inputs.rays.V,
            t,
            inputs.target()[0],
            valid,
            variables_hit=inputs.ray_variables_dict(valid),
            variables_miss=inputs.ray_variables_dict(~valid),
            domain=collective.ray_variables_domains,
        )

        # Render joints
        rendered_joints = tlmviewer.render_joint(inputs.fk.direct)

        return [rendered_surface] + rendered_rays + rendered_joints


class ImagePlaneArtist(Artist):
    def render(self, collective: "Collective", module: nn.Module) -> list[Any]:
        # Render module
        inputs = collective.input_tree[module]
        t, normals, valid, fk_surface, fk_next = collective.output_tree[module.surface]

        # Render surface
        rendered_surface = module.surface.render()
        rendered_surface["matrix"] = fk_surface.direct.tolist()

        # Render rays
        rendered_rays = tlmviewer.render_hit_miss_rays(
            inputs.rays.P,
            inputs.rays.V,
            t,
            inputs.target()[0],
            valid,
            variables_hit=inputs.ray_variables_dict(valid),
            variables_miss=inputs.ray_variables_dict(~valid),
            domain=collective.ray_variables_domains,
        )

        # Render joints
        rendered_joints = tlmviewer.render_joint(inputs.fk.direct)

        return [rendered_surface] + rendered_rays + rendered_joints


class FocalPointArtist(Artist):
    def render(self, collective: "Collective", module: nn.Module) -> list[Any]:
        inputs = collective.input_tree[module]

        # Render module
        target = inputs.target().unsqueeze(0)
        rendered_module = [tlmviewer.render_points(target, "red")]

        # Render rays
        # Distance from ray origin P to target
        dist = torch.linalg.vector_norm(inputs.rays.P - inputs.target(), dim=1)

        # Always draw rays in their positive t direction
        t = torch.abs(dist)
        rendered_rays = tlmviewer.render_rays_length(
            inputs.rays.P,
            inputs.rays.V,
            t,
            layer=tlmviewer.LAYER_VALID_RAYS,
            variables=inputs.ray_variables_dict(),
            domain=collective.ray_variables_domains,
            default_color=tlmviewer.color_valid,
        )

        rendered_joints = tlmviewer.render_joint(inputs.fk.direct)
        return rendered_module + rendered_rays + rendered_joints


class SequentialArtist(Artist):
    def render(self, collective: "Collective", module: nn.Module) -> list[Any]:
        nodes = []
        for child in module.children():
            nodes.extend(collective.render(child))
        return nodes


class KinematicArtist(Artist):
    def render(self, collective: "Collective", module: nn.Module) -> list[Any]:
        tf = collective.input_tree[module]
        return tlmviewer.render_joint(tf.direct)
