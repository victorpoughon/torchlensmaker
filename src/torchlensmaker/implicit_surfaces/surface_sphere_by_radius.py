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

import torch
import torch.nn as nn
import math

from typing import Any
from functools import partial
from jaxtyping import Float
from torchlensmaker.core.geometry import unit_vector, within_radius

from torchlensmaker.types import (
    ScalarTensor,
    BatchTensor,
    Batch2DTensor,
    BatchNDTensor,
    MaskTensor,
    Tf2D,
)

from torchlensmaker.kinematics.homogeneous_geometry import (
    hom_identity_2d,
)

from torchlensmaker.core.functional_kernel import FunctionalKernel
from torchlensmaker.core.tensor_manip import init_param

from .raytrace import raytrace
from .sag_geometry import lens_diameter_domain_2d, anchor_transforms_2d
from .kernels_utils import example_rays_2d


def sphere_radius_center(dim: int, R: ScalarTensor) -> Float[torch.Tensor, " D"]:
    return torch.cat(
        (R.unsqueeze(0), torch.zeros((dim - 1,), dtype=R.dtype, device=R.device))
    )


def sphere_radius_normals(R: ScalarTensor, points: BatchNDTensor) -> MaskTensor:
    batch, dim, dtype = points.shape[:-1], points.shape[-1], R.dtype

    # The normal is the vector from the center to the points
    center = sphere_radius_center(dim, R)
    normals = torch.nn.functional.normalize(points - center, dim=-1)

    # We need a default value for the case where point == center, to avoid div by zero
    unit = unit_vector(dim, dtype)
    normal_at_origin = torch.tile(unit, ((*batch, 1)))

    return torch.where(
        torch.all(torch.isclose(center, points), dim=-1)
        .unsqueeze(-1)
        .expand_as(normals),
        normal_at_origin,
        normals,
    )


def sphere_radius_extent_x(diameter: ScalarTensor, R: ScalarTensor) -> ScalarTensor:
    tau = diameter / 2
    R2, tau2 = R**2, tau**2
    return R - torch.sqrt(R2 - tau2)


def sphere_radius_contains(
    diameter: ScalarTensor, R: ScalarTensor, points: BatchNDTensor, tol: float
) -> MaskTensor:
    center = sphere_radius_center(points.shape[-1], R)
    within_outline = within_radius(diameter / 2 + tol, points)
    on_sphere = (
        torch.abs(torch.linalg.vector_norm(points - center, dim=-1) - torch.abs(R))
        <= tol
    )
    within_extent = (
        torch.abs(points[:, 0]) <= torch.abs(sphere_radius_extent_x(diameter, R)) + tol
    )

    return torch.all(
        torch.stack((within_outline, on_sphere, within_extent), dim=-1), dim=-1
    )


def sphere_radius_raytracing(
    diameter: ScalarTensor, R: ScalarTensor, P: BatchNDTensor, V: BatchNDTensor
) -> tuple[BatchTensor, BatchNDTensor, MaskTensor]:
    N, D = P.shape

    dim, dtype, device = P.shape[-1], P.dtype, P.device

    # For numerical stability, it's best if P is close to the origin
    # Bring rays origins as close as possible before solving
    init_t = -torch.sum(P * V, dim=-1) / torch.sum(V * V, dim=-1)
    P = P + init_t.unsqueeze(-1).expand_as(V) * V

    center = sphere_radius_center(dim, R)

    # Sphere-ray collision is a second order polynomial
    A = torch.sum(V**2, dim=1)
    B = 2 * torch.sum(V * (P - center), dim=1)
    C = torch.sum((P - center) ** 2, dim=1) - R**2
    assert A.shape == B.shape == C.shape == (N,)

    delta = B**2 - 4 * A * C
    safe_delta = torch.clamp(delta, min=0.0)
    assert delta.shape == (N,), delta.shape
    assert safe_delta.shape == (N,), safe_delta.shape

    # tensor of shape (N, 2) with both roots
    # safe meaning that if the root is undefined the value is zero instead
    safe_roots = torch.stack(
        (
            (-B + torch.sqrt(safe_delta)) / (2 * A),
            (-B - torch.sqrt(safe_delta)) / (2 * A),
        ),
        dim=1,
    )
    assert safe_roots.shape == (N, 2)

    # mask of shape (N, 2) indicating if each root is inside the outline
    tol = {torch.float32: 1e-3, torch.float64: 1e-7}[dtype]
    root_inside = torch.stack(
        (
            sphere_radius_contains(
                diameter, R, P + safe_roots[:, 0].unsqueeze(1).expand_as(V) * V, tol
            ),
            sphere_radius_contains(
                diameter, R, P + safe_roots[:, 1].unsqueeze(1).expand_as(V) * V, tol
            ),
        ),
        dim=1,
    )
    assert root_inside.shape == (N, 2)

    # number of valid roots
    number_of_valid_roots = torch.sum(root_inside, dim=1)
    assert number_of_valid_roots.shape == (N,)

    # index of the first valid root
    _, index_first_valid = torch.max(root_inside, dim=1)
    assert index_first_valid.shape == (N,)

    # index of the root closest to zero
    _, index_closest = torch.min(torch.abs(safe_roots), dim=1)
    assert index_closest.shape == (N,)

    # delta < 0 => no collision
    # delta >=0 => two roots (which maybe equal)
    #  - if both are outside the outline => no collision
    #  - if only one is inside the outline => one collision
    #  - if both are inside the outline => return the root closest to zero (i.e. the ray origin)

    # TODO refactor don't rely on contains at all

    default_t = torch.zeros(N, dtype=dtype)
    arange = torch.arange(N)
    t = torch.where(
        delta < 0,
        default_t,
        torch.where(
            number_of_valid_roots == 0,
            default_t,
            torch.where(
                number_of_valid_roots == 1,
                safe_roots[arange, index_first_valid],
                torch.where(
                    number_of_valid_roots == 2,
                    safe_roots[arange, index_closest],
                    default_t,
                ),
            ),
        ),
    )

    local_points = P + t.unsqueeze(1).expand_as(V) * V
    local_normals = sphere_radius_normals(R, local_points)
    valid = number_of_valid_roots > 0

    return init_t + t, local_normals, valid


class SphereByRadius2DSurfaceKernel(FunctionalKernel):
    """
    Functional kernel for a 2D spherical arc parameterized by:
        - signed surface radius
        - lens diameter

    with support for anchors and scale.
    """

    inputs = {
        "P": Batch2DTensor,
        "V": Batch2DTensor,
        "tf_in": Tf2D,
    }

    params = {
        "diameter": ScalarTensor,
        "R": ScalarTensor,
        "anchors": Float[torch.Tensor, " 2"],
        "scale": ScalarTensor,
    }

    outputs = {
        "t": BatchTensor,
        "normals": Batch2DTensor,
        "valid": MaskTensor,
        "surface_tf": Tf2D,
        "next_tf": Tf2D,
    }

    def apply(
        self,
        P: Batch2DTensor,
        V: Batch2DTensor,
        tf_in: Tf2D,
        diameter: ScalarTensor,
        R: ScalarTensor,
        anchors: Float[torch.Tensor, " 2"],
        scale: ScalarTensor,
    ) -> tuple[BatchTensor, Batch2DTensor, MaskTensor, Tf2D, Tf2D]:
        # Setup the local solver for this surface class
        local_solver = partial(sphere_radius_raytracing, diameter=diameter, R=R)

        # Compute anchor transforms from anchors and scale
        extent = sphere_radius_extent_x(diameter, R)
        tf_surface, tf_next = anchor_transforms_2d(
            anchors[0] * extent, anchors[1] * extent, scale, tf_in
        )

        # Perform raytrace
        t, normals, valid = raytrace(P, V, tf_surface, local_solver)

        return t, normals, valid, tf_surface, tf_next

    def example_inputs(
        self, dtype: torch.dtype, device: torch.device
    ) -> tuple[Batch2DTensor, Batch2DTensor, Tf2D]:
        P, V = example_rays_2d(10, dtype, device)
        tf = hom_identity_2d(dtype, device)
        return P, V, tf

    def example_params(
        self, dtype: torch.dtype, device: torch.device
    ) -> tuple[ScalarTensor, ScalarTensor, Float[torch.Tensor, " 2"], ScalarTensor]:
        return (
            torch.tensor(10.0, dtype=dtype, device=device),
            torch.tensor(0.5, dtype=dtype, device=device),
            torch.tensor((0.0, 0.0), dtype=dtype, device=device),
            torch.tensor(-1.0, dtype=dtype, device=device),
        )


# TODO 3D


class SphereByRadius(nn.Module):
    """
    Spherical surface (2D or 3D) parameterized by lens diameter and radius.
    """

    def __init__(
        self,
        diameter: float | ScalarTensor,
        R: float | ScalarTensor | nn.Parameter,
        *,
        anchors: tuple[float, float] | Float[torch.Tensor, " 2"] = (0.0, 0.0),
        scale: float | ScalarTensor = 1.0,
        trainable: bool = True,
    ):
        super().__init__()
        self.diameter = init_param(self, "diameter", diameter, False)
        self.R = init_param(self, "R", R, trainable)
        self.anchors = init_param(self, "anchors", anchors, False)
        self.scale = init_param(self, "scale", scale, False)
        self.func2d = SphereByRadius2DSurfaceKernel()

    def forward(
        self, P: BatchTensor, V: BatchTensor, tf: Tf2D
    ) -> tuple[BatchTensor, BatchNDTensor, MaskTensor, Tf2D, Tf2D]:
        return self.func2d.apply(
            P, V, tf, self.diameter, self.R, self.anchors, self.scale
        )

    def render(self) -> Any:
        return {
            "type": "surface-sphere-r",
            "diameter": self.diameter.item(),
            "R": self.R.item(),
        }
