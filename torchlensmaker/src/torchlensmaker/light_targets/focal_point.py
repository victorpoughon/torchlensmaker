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

from typing import Any, Self

import torch

from torchlensmaker.core.ray_bundle import RayBundle
from torchlensmaker.kinematics.homogeneous_geometry import hom_target
from torchlensmaker.light_targets.light_target import LightTarget, LightTargetOutput
from torchlensmaker.sequential.model_trace import ModelTrace
from torchlensmaker.sequential.sequential_data import SequentialData
from torchlensmaker.surfaces import SurfaceElementOutput
from torchlensmaker.types import Tf


class FocalPoint(LightTarget):
    def __init__(self) -> None:
        super().__init__()

    def clone(self, **overrides: Any) -> Self:
        return type(self)()

    def forward(self, rays: RayBundle, tf: Tf) -> LightTargetOutput:
        dim = rays.P.shape[-1]
        N = rays.P.shape[0]

        X = hom_target(tf.direct)
        P = rays.P
        V = rays.V

        # Compute ray-point squared distance

        # If 2D, pad to 3D with zeros
        if dim == 2:
            X = torch.cat((X, torch.zeros(1)), dim=0)
            P = torch.cat((P, torch.zeros((N, 1))), dim=1)
            V = torch.cat((V, torch.zeros((N, 1))), dim=1)

        cross = torch.cross(X - P, V, dim=1)
        norm = torch.norm(V, dim=1)

        distance = torch.norm(cross, dim=1) / norm

        loss = distance.sum() / N

        return LightTargetOutput(
            loss=loss,
            surface_outputs=SurfaceElementOutput(
                t=None,
                normals=None,
                valid=None,
                points_local=None,
                points_global=None,
                rsm=None,
                tf_surface=tf,
                tf_next=tf,
            ),  # TODO probably need to use an actual surface for focal point, plane?
        )

    def sequential(self, data: SequentialData) -> SequentialData:
        out = self(data.rays, data.fk)
        # In sequential mode, light targets are transparent to rays
        # We evaluate the optical element outputs but forward the data unchanged
        return data

    def trace(self, trace: ModelTrace, key: str, inputs: Any, outputs: Any) -> None:
        input_rays, input_tf = inputs
        trace.add_input_joint(key, input_tf)
        trace.add_output_joint(key, outputs.surface_outputs.tf_next)
        trace.add_input_rays(key, input_rays)
        trace.add_focal_point(key, input_tf)
