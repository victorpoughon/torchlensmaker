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

from jaxtyping import Int, Float
import torch


def disk_sampling(
    Nrho: Int[torch.Tensor, ""],
    Ntheta: Int[torch.Tensor, ""],
    dtype: torch.dtype,
    device: torch.device,
) -> Float[torch.Tensor, "N 2"]:
    "Samples 2D points on a disk"

    y = torch.linspace(0, 1, Nrho, dtype=dtype, device=device).to(dtype=dtype)
    x = torch.linspace(0, 1, Ntheta, dtype=dtype, device=device).to(dtype=dtype)
    xx, yy = torch.meshgrid((x, y), indexing="ij")

    rho = torch.sqrt(yy)
    theta = 2 * torch.tensor(torch.pi, dtype=dtype) * xx

    X = rho * torch.cos(theta)
    Y = rho * torch.sin(theta)

    return torch.stack((X.reshape(-1), Y.reshape(-1)), dim=-1)


def spiral_disk_sampling(
    Nrho: Int[torch.Tensor, ""],
    Ntheta: Int[torch.Tensor, ""],
    spiral_coefficent: Float[torch.Tensor, ""],
    dtype: torch.dtype,
    device: torch.device,
) -> Float[torch.Tensor, "N 2"]:
    "Samples 2D points on a disk"

    y = torch.linspace(0, 1, Nrho, dtype=dtype, device=device)
    x = torch.linspace(0, 1, Ntheta, dtype=dtype, device=device)
    xx, yy = torch.meshgrid((x, y), indexing="ij")

    rho = torch.sqrt(yy)
    theta = torch.remainder(
        2 * torch.pi * xx + spiral_coefficent * rho * 2 * torch.pi, 2 * torch.pi
    )

    X = rho * torch.cos(theta)
    Y = rho * torch.sin(theta)

    return X.reshape(-1), Y.reshape(-1)
