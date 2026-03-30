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


from typing import Any

from torchlensmaker.core.base_module import BaseModule
from torchlensmaker.core.ray_bundle import RayBundle
from torchlensmaker.sequential.model_trace import ModelTrace
from torchlensmaker.sequential.sequential_data import SequentialData
from torchlensmaker.surfaces.surface_element import SurfaceElementOutput
from torchlensmaker.types import Tf


class OpticalSurfaceElement(BaseModule):
    def forward(
        self, rays: RayBundle, tf: Tf
    ) -> tuple[RayBundle, SurfaceElementOutput]:
        raise NotImplementedError

    def sequential(self, data: SequentialData) -> SequentialData:
        new_rays, surface_outputs = self(data.rays, data.fk)
        return data.replace(rays=new_rays, fk=surface_outputs.tf_next)

    def trace(self, trace: ModelTrace, key: str, inputs: Any, outputs: Any) -> None:
        input_rays, input_tf = inputs
        new_rays, surface_outputs = outputs
        trace.add_input_joint(key, input_tf)
        trace.add_output_joint(key, surface_outputs.tf_next)
        trace.add_input_rays(key, input_rays)
        trace.add_output_rays(key, new_rays)
        trace.add_surface(key, (surface_outputs.tf_surface, self.surface))
        trace.add_collision(
            key, surface_outputs.t, surface_outputs.normals, surface_outputs.valid
        )
