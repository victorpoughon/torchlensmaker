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

import torch
import torch.nn as nn

from torchlensmaker.core.base_module import BaseModule
from torchlensmaker.core.ray_bundle import RayBundle
from torchlensmaker.core.tensor_manip import filter_optional_mask
from torchlensmaker.kinematics.homogeneous_geometry import hom_target
from torchlensmaker.light_sources.light_sources_elements import LightSourceBase
from torchlensmaker.sequential.model_trace import ModelTrace
from torchlensmaker.sequential.utils import get_elements_by_type
from torchlensmaker.types import MaskTensor
from torchlensmaker.viewer import tlmviewer


def raybundle_var_dict(rays: RayBundle) -> dict[str, torch.Tensor]:
    "RayBundle to tlmviewer variable dict"
    dim = rays.P.shape[-1]
    d = {}

    def add(name: str, t: torch.Tensor):
        d[name] = t

    if dim == 2:
        add("pupil", rays.pupil)
        add("field", rays.field)
    if dim == 3:
        add("pupil0", rays.pupil[:, 0])
        add("pupil1", rays.pupil[:, 1])
        add("field0", rays.field[:, 0])
        add("field1", rays.field[:, 1])

    add("wavelength", rays.wavel)
    add("source", rays.source_idx)

    return d


def domain_union(a: dict[str, list[float]], b: dict[str, list[float]]):
    U = dict(a)
    # merge b into a
    for key, val in b.items():
        if key not in a:
            U[key] = val
        else:
            amin, amax = U[key]
            bmin, bmax = val
            U[key] = [min(amin, bmin), max(amax, bmax)]
    return U


def get_domain(optics: nn.Module, dim: int) -> dict[str, list[float]]:
    light_sources = get_elements_by_type(optics, LightSourceBase)

    if len(light_sources) == 0:
        return {}

    domain = {}

    # Compute union of all light sources domains
    for ls in light_sources:
        ls = cast(LightSourceBase, ls)
        d = ls.domain(dim)
        domain = domain_union(domain, d)

    return domain


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

        # if the input rays have a matching collision entry, use it to render rays as hit / miss
        if key in trace.collisions:
            t, normals, valid = trace.collisions[key]

            ret.extend(
                tlmviewer.render_hit_miss_rays(
                    input_rays.P,
                    input_rays.V,
                    t,
                    hom_target(input_tf.direct)[0],
                    valid,
                    variables_hit=raybundle_var_dict(input_rays.mask(valid)),
                    variables_miss=raybundle_var_dict(input_rays.mask(~valid)),
                    domain=domain,
                )
            )
        else:
            # else render rays until the joint
            target = hom_target(input_tf.direct)
            dist = torch.linalg.vector_norm(input_rays.P - target, dim=1)
            # Always draw rays in their positive t direction
            t = torch.abs(dist)
            ret.extend(
                tlmviewer.render_rays_length(
                    input_rays.P,
                    input_rays.V,
                    t,
                    category=tlmviewer.CATEGORY_VALID_RAYS,
                    variables=raybundle_var_dict(input_rays),
                    domain=domain,
                    default_color=tlmviewer.color_valid,
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
        variables=raybundle_var_dict(rays),
        domain=domain,
        default_color=tlmviewer.color_valid,
        category=tlmviewer.CATEGORY_OUTPUT_RAYS,
    )


def trace_render_focal_points(trace: ModelTrace) -> list[Any]:
    ret = []
    for key, fp in trace.focal_points.items():
        target = hom_target(fp.direct).unsqueeze(0)
        ret.append(tlmviewer.render_points(target, "red"))

    return ret


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

    # Render end rays
    viewer_scene["data"].extend(
        trace_render_end_rays(trace, end, ray_variables_domains)
    )

    # Render focal point
    viewer_scene["data"].extend(trace_render_focal_points(trace))

    return viewer_scene
