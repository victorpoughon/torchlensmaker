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
from torchlensmaker.sequential.optical_trace import OpticalTrace
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

    def trace(self, trace: OpticalTrace, key: str, inputs: Any, outputs: Any) -> None:
        record = outputs
        trace.add_node(
            key=key,
            record=record,
            module=self,
            new_bundle=record.output_rays,
            new_tf=record.surface_record.tf_next,
        )
