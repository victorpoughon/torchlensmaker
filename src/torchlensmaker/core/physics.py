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
from torch.nn.functional import normalize

from typing import Literal

Tensor = torch.Tensor
RefractionCriticalAngleMode = Literal["drop", "nan", "clamp", "reflect"]


def reflection(rays: Tensor, normals: Tensor) -> Tensor:
    """
    Vector based reflection.

    Args:
        ray: unit vectors of the incident rays, shape (B, 2)
        normal: unit vectors normal to the surface, shape (B, 2)

    Returns:
        vectors of the reflected vector with shape (B, 2)
    """

    dot_product = torch.sum(rays * normals, dim=1, keepdim=True)
    R = rays - 2 * dot_product * normals
    return torch.div(R, torch.norm(R, dim=1, keepdim=True))


def refraction(
    rays: Tensor,
    normals: Tensor,
    n1: float | Tensor,
    n2: float | Tensor,
    critical_angle: RefractionCriticalAngleMode = "drop",
) -> Tensor:
    """
    Vector based refraction (Snell's law).

    The 'critical_angle' argument specifies how incident rays beyond the
    critical angle are handled:

        * 'nan': Incident rays beyond the critical angle will refract
          as nan values. The returned tensor always has the same shape as the
          input tensors.

        * 'clamp': Incident rays beyond the critical angle all refract at 90°.
          The returned tensor always has the same shape as the input tensors.

        * 'drop' (default): Incident rays beyond the critical angle will not be
          refracted. The returned tensor doesn't necesarily have the same shape
          as the input tensors.

        * 'reflect': Incident rays beyond the critical angle are reflected. This
          is true physical behavior aka total internal reflection. The returned
          tensor always has the same shape as the input tensors.

    Args:
        rays: unit vectors of the incident rays, shape (N, 2/3)
        normals: unit vectors normal to the surface, shape (N, 2/3)
        n1: index of refraction of the incident medium, float or tensor of shape (N)
        n2: index of refraction of the refracted medium float or tensor of shape (N)
        critical_angle: one of 'nan', 'clamp', 'drop', 'reflect' (default: 'nan')

    Returns:
        unit vectors of the refracted rays, shape (C, 2)
        valid: boolean teansor of shape (N,) indicating which rays are not
               beyond the critical angle
    """

    assert rays.dim() == 2 and rays.shape[1] in {2, 3}
    assert normals.dim() == 2 and normals.shape[1] in {2, 3}
    assert rays.shape[0] == normals.shape[0]

    # Compute dot product for the batch, aka cosine of the incident angle
    cos_theta_i = torch.sum(rays * -normals, dim=1, keepdim=True)

    # Convert n1 and n2 into tensors
    N = rays.shape[0]
    n1 = torch.as_tensor(n1).expand((N,))
    n2 = torch.as_tensor(n2).expand((N,))

    # Compute R_perp
    eta = n1 / n2
    R_perp = eta.unsqueeze(1) * (rays + cos_theta_i * normals)

    # Radicand of R_para and critical angle mask
    radicand = 1 - torch.sum(R_perp * R_perp, dim=1, keepdim=True)
    valid = (radicand >= 0.0).squeeze(1)

    # Compute R_para, depending on critical angle option
    if critical_angle == "nan":
        R_para = -torch.sqrt(radicand) * normals

        return normalize(R_perp + R_para), valid

    elif critical_angle == "clamp":
        radicand = torch.clamp(radicand, min=0.0, max=None)
        R_para = -torch.sqrt(radicand) * normals

        return normalize(R_perp + R_para), valid

    elif critical_angle == "drop":
        R_para = -torch.sqrt(radicand[valid, :]) * normals[valid, :]
        R_perp = R_perp[valid, :]

        return normalize(R_perp + R_para), valid

    elif critical_angle == "reflect":
        radicand = 1 - torch.sum(R_perp * R_perp, dim=1, keepdim=True)
        valid = (radicand >= 0.0).squeeze(1)
        R_para = (
            -torch.sqrt(1 - torch.sum(R_perp * R_perp, dim=1, keepdim=True)) * normals
        )
        R = R_perp + R_para

        R[~valid] = reflection(rays, normals)[~valid]
        return normalize(R), valid

    else:
        raise ValueError(
            f"critical_angle must be one of 'nan', 'clamp', 'drop', 'reflect'. Got {repr(critical_angle)}."
        )
