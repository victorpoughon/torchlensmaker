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
from torchlensmaker.kinematics.homogeneous_geometry import hom_target
from torchlensmaker.light_sources.light_sources_elements import LightSourceBase
from torchlensmaker.light_targets.light_target import LightTargetRecord
from torchlensmaker.optical_surfaces.optical_surface import OpticalSurfaceRecord
from torchlensmaker.sequential.optical_trace import OpticalTrace
from torchlensmaker.sequential.utils import get_elements_by_type
from torchlensmaker.viewer import tlmviewer


def raybundle_var_dict(rays: RayBundle) -> dict[str, torch.Tensor]:
    "RayBundle to tlmviewer variable dict"
    dim = rays.P.shape[-1]
    d = {}

    def add(name: str, t: torch.Tensor):
        d[name] = t

    if dim == 2:
        add("pupil", rays.pupil.values)
        add("field", rays.field.values)
    if dim == 3:
        add("pupil0", rays.pupil.values[:, 0])
        add("pupil1", rays.pupil.values[:, 1])
        add("field0", rays.field.values[:, 0])
        add("field1", rays.field.values[:, 1])

    add("wavelength", rays.wavel.values)
    add("source", rays.source.idx)

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


def trace_render_surfaces(trace: OpticalTrace) -> list[Any]:
    surfaces = []
    for key, node in trace.iter_nodes_by_record_type(OpticalSurfaceRecord):
        surface = node.module.surface
        tf = node.record.surface_record.tf_surface
        surf = tlmviewer.render_surface(surface, tf.direct, dim=trace.dim)
        if surf is not None:
            surfaces.append(surf)

    return surfaces


def trace_render_joints(trace: OpticalTrace) -> list[Any]:
    ret = []
    for node in trace.nodes.values():
        tf = node.tf_out
        ret.extend(tlmviewer.render_joint(tf.direct))
    return ret


def trace_render_rays(trace: OpticalTrace, domain: dict[str, list[float]]) -> list[Any]:
    ret = []

    for key, node in trace.iter_nodes_by_record_type(OpticalSurfaceRecord):
        input_tf = node.tf_in
        input_rays = node.bundle_in

        sr = node.record.surface_record
        t, normals, collision_valid = sr.t, sr.normals, sr.valid

        rays_valid_coll = input_rays.filter(input_rays.valid & collision_valid)

        # Render hit rays
        ret.append(
            tlmviewer.render_rays(
                rays_valid_coll.P,
                rays_valid_coll.points_at(t[input_rays.valid & collision_valid]),
                variables=raybundle_var_dict(rays_valid_coll),
                domain=domain,
                default_color=tlmviewer.color_valid,
                category=tlmviewer.CATEGORY_VALID_RAYS,
            )
        )

        # Render miss rays: rays absorbed because not colliding
        miss_mask = input_rays.valid & ~collision_valid
        if miss_mask.sum() > 0:
            ret.extend(
                tlmviewer.render_rays_misses(
                    input_rays.P[miss_mask],
                    input_rays.V[miss_mask],
                    hom_target(input_tf.direct)[0],
                    variables=raybundle_var_dict(input_rays.filter(miss_mask)),
                    domain=domain,
                    default_color=tlmviewer.color_blocked,
                    category=tlmviewer.CATEGORY_BLOCKED_RAYS,
                )
            )

        # old code for rendering rays until focal point:
        # else:
        #     # else render rays until the joint
        #     target = hom_target(input_tf.direct)
        #     dist = torch.linalg.vector_norm(bundle_valid.P - target, dim=1)
        #     # Always draw rays in their positive t direction
        #     t = torch.abs(dist)
        #     ret.extend(
        #         tlmviewer.render_rays_length(
        #             bundle_valid.P,
        #             bundle_valid.V,
        #             t,
        #             category=tlmviewer.CATEGORY_VALID_RAYS,
        #             variables=raybundle_var_dict(bundle_valid),
        #             domain=domain,
        #             default_color=tlmviewer.color_valid,
        #         )
        #     )

    return ret


def trace_render_end_rays(
    trace: OpticalTrace, end: float | None, domain: dict[str, list[float]]
) -> list[Any]:
    if end is None:
        return []

    _, node = next(reversed(trace.nodes.items()))
    rays = node.bundle_out.filter(node.bundle_out.valid)
    return tlmviewer.render_rays_length(
        rays.P,
        rays.V,
        end,
        variables=raybundle_var_dict(rays),
        domain=domain,
        default_color=tlmviewer.color_valid,
        category=tlmviewer.CATEGORY_OUTPUT_RAYS,
    )


def trace_render_focal_points(trace: OpticalTrace) -> list[Any]:
    ret = []
    for key, node in trace.iter_nodes_by_record_type(LightTargetRecord):
        target = hom_target(node.tf_out.direct).unsqueeze(0)
        ret.append(tlmviewer.render_points(target, "red"))

    return ret


def render_model_trace(
    model: BaseModule,
    trace: OpticalTrace,
    end: float | None = None,
) -> Any:
    "Render an OpticalScene to a tlmviewer scene"

    dim = trace.dim
    viewer_scene = tlmviewer.new_scene("2D" if dim == 2 else "3D")

    # Figure out available ray variables and their range,
    # this will be used for rays coloring info by tlmviewer
    ray_variables_domains = get_domain(model, dim)

    # Render parts of the scene: surfaces, joints, rays
    viewer_scene.data.extend(trace_render_surfaces(trace))
    viewer_scene.data.extend(trace_render_joints(trace))
    viewer_scene.data.extend(trace_render_rays(trace, ray_variables_domains))

    # Render end rays
    viewer_scene.data.extend(trace_render_end_rays(trace, end, ray_variables_domains))

    # Render focal point
    viewer_scene.data.extend(trace_render_focal_points(trace))

    return viewer_scene
