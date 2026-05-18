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


from dataclasses import dataclass
from typing import Any

from torchlensmaker.core.base_module import BaseModule
from torchlensmaker.core.ray_bundle import RayBundle
from torchlensmaker.sequential.model_trace import ModelTrace
from torchlensmaker.sequential.sequential_data import SequentialData
from torchlensmaker.surfaces import SurfaceRecord
from torchlensmaker.types import Tf


@dataclass
class OpticalSurfaceRecord:
    output_rays: RayBundle
    surface_record: SurfaceRecord


class OpticalSurfaceElement(BaseModule):
    def forward(self, rays: RayBundle, tf: Tf) -> OpticalSurfaceRecord:
        raise NotImplementedError

    def sequential(self, data: SequentialData) -> SequentialData:
        record = self(data.rays, data.fk)
        return data.replace(rays=record.output_rays, fk=record.surface_record.tf_next)

    def trace(self, trace: ModelTrace, key: str, inputs: Any, outputs: Any) -> None:
        input_rays, input_tf = inputs
        record = outputs
        trace.add_input_joint(key, input_tf)
        trace.add_output_joint(key, record.surface_record.tf_next)
        trace.add_input_rays(key, input_rays)
        trace.add_output_rays(key, record.output_rays)
        trace.add_surface(key, (record.surface_record.tf_surface, self.surface))
        trace.add_collision(
            key,
            record.surface_record.t,
            record.surface_record.normals,
            record.surface_record.valid,
        )
