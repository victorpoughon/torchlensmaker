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

from typing import Any, Callable, Optional

from torchlensmaker.core.ray_bundle import RayBundle
from torchlensmaker.kinematics.homogeneous_geometry import (
    kinematic_chain_append,
    transform_points,
    hom_target,
)
from torchlensmaker.core.tensor_manip import (
    filter_optional_mask,
)

from . import tlmviewer

from .rendering import Collective
from .rendering import Artist

Tensor = torch.Tensor

# TODO remove
def ray_variables_dict(
    rays: RayBundle, valid: Optional[torch.Tensor] = None
) -> dict[str, torch.Tensor]:
    "Convert ray variables from to a dict of Tensors"
    d = {}

    def update(tensor: Optional[torch.Tensor], name: str) -> None:
        # TODO this if check is temporary to avoid a divide by zero in tlmviewer
        # ideally we would export all three variables allways, and tlmviewer
        # handles correctly degenerate cases like PointSource which has all
        # field coord = 0, or a single wavelength, etc.
        if tensor.numel() > 0 and (tensor.max() - tensor.min()) > 1e-3:
            d[name] = filter_optional_mask(tensor, valid)

    # TODO no support for 2D colormaps in tlmviewer yet
    # but base and object are 2D variables in 3D
    # TODO tlmviewer: rename base/object to pupil/field
    dim = rays.P.shape[-1]
    if dim == 2:
        update(rays.pupil, "base")
        update(rays.field, "object")

    update(rays.wavel, "wavelength")

    return d

class ForwardArtist(Artist):
    "Forward rendering to a subobject"

    def __init__(self, getter: Callable[[nn.Module], nn.Module]):
        super().__init__()
        self.getter = getter

    def render(self, collective: "Collective", module: nn.Module) -> list[Any]:
        return collective.render(self.getter(module))


class SurfacePropagatorArtist(Artist):
    def render(self, collective: "Collective", module: nn.Module) -> list[Any]:
        # Render module
        input_rays, input_tf = collective.input_tree[module]
        t, normals, valid, fk_surface, fk_next = collective.output_tree[module.surface]

        # Render surface
        rendered_surface = module.surface.render()
        rendered_surface["matrix"] = fk_surface.direct.tolist()

        # Render rays
        rendered_rays = tlmviewer.render_hit_miss_rays(
            input_rays.P,
            input_rays.V,
            t,
            hom_target(input_tf.direct)[0],
            valid,
            variables_hit=ray_variables_dict(input_rays, valid),
            variables_miss=ray_variables_dict(input_rays, ~valid),
            domain=collective.ray_variables_domains,
        )

        # Render joints
        rendered_joints = tlmviewer.render_joint(input_tf.direct)

        return [rendered_surface] + rendered_rays + rendered_joints



class FocalPointArtist(Artist):
    def render(self, collective: "Collective", module: nn.Module) -> list[Any]:
        inputs = collective.input_tree[module]

        # Render module
        target = inputs.target().unsqueeze(0)
        rendered_module = [tlmviewer.render_points(target, "red")]

        # Render rays
        # Distance from ray origin P to target
        target = hom_target(inputs.fk.direct)
        dist = torch.linalg.vector_norm(inputs.rays.P - target, dim=1)

        # Always draw rays in their positive t direction
        t = torch.abs(dist)
        rendered_rays = tlmviewer.render_rays_length(
            inputs.rays.P,
            inputs.rays.V,
            t,
            layer=tlmviewer.LAYER_VALID_RAYS,
            variables=ray_variables_dict(inputs.rays),
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
