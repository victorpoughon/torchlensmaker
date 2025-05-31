# This file is part of Torch Lens Maker
# Copyright (C) 2025 Victor Poughon
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


from torchlensmaker.elements.sequential import SequentialElement
from torchlensmaker.optical_data import OpticalData


Tensor = torch.Tensor


class FocalPoint(SequentialElement):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs: OpticalData) -> OpticalData:
        dim = inputs.dim
        N = inputs.P.shape[0]

        X = inputs.target()
        P = inputs.P
        V = inputs.V

        # Compute ray-point squared distance distance

        # If 2D, pad to 3D with zeros
        if dim == 2:
            X = torch.cat((X, torch.zeros(1)), dim=0)
            P = torch.cat((P, torch.zeros((N, 1))), dim=1)
            V = torch.cat((V, torch.zeros((N, 1))), dim=1)

        cross = torch.cross(X - P, V, dim=1)
        norm = torch.norm(V, dim=1)

        distance = torch.norm(cross, dim=1) / norm

        loss = distance.sum() / N

        return inputs.replace(loss=inputs.loss + loss)
