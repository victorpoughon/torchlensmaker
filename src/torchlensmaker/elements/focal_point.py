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
from torchlensmaker.elements.sequential import SequentialElement
from torchlensmaker.elements.sequential_data import SequentialData
from torchlensmaker.kinematics.homogeneous_geometry import hom_target
from torchlensmaker.types import BatchNDTensor, Direction, ScalarTensor, Tf


class FocalPoint(SequentialElement):
    def __init__(self) -> None:
        super().__init__()

    def clone(self, **overrides: Any) -> Self:
        return type(self)()

    def sequential(self, inputs: SequentialData) -> SequentialData:
        _, _ = self(inputs.rays, inputs.fk, inputs.direction)
        return inputs

    def forward(
        self, rays: RayBundle, tf: Tf, direction: Direction
    ) -> tuple[BatchNDTensor, ScalarTensor]:
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

        rays_image = torch.zeros((), dtype=rays.dtype, device=rays.device)
        return rays_image, loss
