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

from typing import Any, cast

import torch.nn as nn

from torchlensmaker.core.base_module import BaseModule
from torchlensmaker.kinematics.homogeneous_geometry import hom_target
from torchlensmaker.light_sources.light_sources_elements import LightSourceBase
from torchlensmaker.sequential.sequential import ModelTrace
from torchlensmaker.sequential.utils import get_elements_by_type
from torchlensmaker.viewer import tlmviewer
from torchlensmaker.viewer.artists import ray_variables_dict


def get_domain(optics: nn.Module, dim: int) -> dict[str, list[float]]:
    light_sources = get_elements_by_type(optics, LightSourceBase)

    if len(light_sources) == 0:
        return {}

    # TODO handle multiple light sources
    ls = cast(LightSourceBase, light_sources[0])

    return ls.domain(dim)


def trace_render_surfaces(trace: ModelTrace) -> list[Any]:
    surfaces = []
    for key, (tf, surface) in trace.surfaces.items():
        surf = surface.render()
        surf["matrix"] = tf.direct.tolist()
        surfaces.append(surf)

    return surfaces


def trace_render_joints(trace: ModelTrace) -> list[Any]:
    ret = []
    for tf in trace.output_joints.values():
        ret.extend(tlmviewer.render_joint(tf.direct))
    return ret


def trace_render_rays(trace: ModelTrace, domain: dict[str, list[float]]) -> list[Any]:
    ret = []

    for key, input_rays in trace.input_rays.items():
        input_tf = trace.input_joints[key]
        t, normals, valid = trace.collisions[key]

        ret.extend(
            tlmviewer.render_hit_miss_rays(
                input_rays.P,
                input_rays.V,
                t,
                hom_target(input_tf.direct)[0],
                valid,
                variables_hit=ray_variables_dict(input_rays, valid),
                variables_miss=ray_variables_dict(input_rays, ~valid),
                domain=domain,
            )
        )

    return ret


def trace_render_end_rays(
    trace: ModelTrace, end: float | None, domain: dict[str, list[float]]
) -> list[Any]:
    if end is None:
        return []

    rays = next(reversed(trace.output_rays.values()))
    return tlmviewer.render_rays_length(
        rays.P,
        rays.V,
        end,
        variables=ray_variables_dict(rays),
        domain=domain,
        default_color=tlmviewer.color_valid,
        layer=tlmviewer.LAYER_OUTPUT_RAYS,
    )


def render_model_trace(
    model: BaseModule,
    trace: ModelTrace,
    end: float | None = None,
) -> Any:
    "Render an OpticalScene to a tlmviewer scene"

    dim = trace.dim
    viewer_scene = tlmviewer.new_scene("2D" if dim == 2 else "3D")

    # Figure out available ray variables and their range,
    # this will be used for rays coloring info by tlmviewer
    ray_variables_domains = get_domain(model, dim)

    # Render parts of the scene: surfaces, joints, rays
    viewer_scene["data"].extend(trace_render_surfaces(trace))
    viewer_scene["data"].extend(trace_render_joints(trace))
    viewer_scene["data"].extend(trace_render_rays(trace, ray_variables_domains))

    # Render end rays, i.e. rays that
    viewer_scene["data"].extend(
        trace_render_end_rays(trace, end, ray_variables_domains)
    )

    return viewer_scene
