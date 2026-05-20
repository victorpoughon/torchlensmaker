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
from torchlensmaker.light_targets.light_target import LightTarget, LightTargetRecord
from torchlensmaker.sequential.optical_trace import OpticalTrace
from torchlensmaker.surfaces import SurfaceRecord
from torchlensmaker.types import Tf


class FocalPoint(LightTarget):
    def __init__(self) -> None:
        super().__init__()

    def clone(self, **overrides: Any) -> Self:
        return type(self)()

    def forward(self, rays: RayBundle, tf: Tf) -> LightTargetRecord:
        dim = rays.P.shape[-1]
        dtype, device = rays.dtype, rays.device

        X = hom_target(tf.direct)
        P = rays.P[rays.valid]
        V = rays.V[rays.valid]
        N = rays.valid.sum()
        Nint: int = int(N.item())

        # If there are no rays, return a constant (non differentiable) loss of zero
        if Nint == 0:
            return LightTargetRecord(
                loss=torch.zeros((), dtype=dtype, device=device),
                surface_outputs=SurfaceRecord(
                    t=None,
                    normals=None,
                    valid=None,
                    points_local=None,
                    points_global=None,
                    rsm=None,
                    tf_surface=tf,
                    tf_next=tf,
                ),
            )

        # Compute ray-point squared distance

        # If 2D, pad to 3D with zeros
        if dim == 2:
            X = torch.cat((X, torch.zeros(1)), dim=0)
            P = torch.cat((P, torch.zeros(Nint, 1)), dim=1)
            V = torch.cat((V, torch.zeros(Nint, 1)), dim=1)

        cross = torch.cross(X - P, V, dim=1)
        norm = torch.norm(V, dim=1)

        distance = torch.norm(cross, dim=1) / norm

        loss = distance.sum() / N

        return LightTargetRecord(
            loss=loss,
            surface_outputs=SurfaceRecord(
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

    def trace(self, trace: OpticalTrace, key: str, upstream_key: str) -> None:
        upstream = trace.nodes[upstream_key]
        record = self(upstream.bundle_out, upstream.tf_out)
        trace.append(
            key=key,
            record=record,
            module=self,
            upstream={upstream_key},
            bundle_in=upstream.bundle_out,
            tf_in=upstream.tf_out,
            new_bundle=None,
            new_tf=None,
        )
