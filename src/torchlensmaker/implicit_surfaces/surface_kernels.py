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

from functools import partial
from typing import Any

from jaxtyping import Float, Int, Bool
import torch


from torchlensmaker.types import (
    ScalarTensor,
    BatchTensor,
    Batch2DTensor,
    Batch3DTensor,
    BatchNDTensor,
    HomMatrix,
)

from torchlensmaker.kinematics.homogeneous_geometry import hom_identity_2d


from torchlensmaker.core.functional_kernel import FunctionalKernel
from .sag_functions import spherical_sag_2d

from .sag_raytrace import sag_surface_local_raytrace_2d, raytrace

# PHASE 1
# implement basic sphere2D kernel
# implement basic sphere2D element
# test sphere2D element in notebook with rays and tlmviewer export
# test onnx export of basic sphere2D kernel

# PHASE 2
# add misssing features:
# TODO lens diameter parameter
# TODO scale and anchor
# TODO add domain to raytrace, compute collision mask

# PHASE 3
# add other sag kernels / elements
# add other implicit surfaces (non sag)
# add explicit surfaces (plane, sphereR)


def example_rays_2d(
    N: int, dtype: torch.dtype, device: torch.device
) -> tuple[BatchNDTensor, BatchNDTensor]:
    P = torch.stack(
        (
            torch.zeros((N,), dtype=dtype, device=device),
            torch.linspace(-1, 1, N, dtype=dtype, device=device),
        ),
        dim=-1,
    )

    V = torch.tensor([[1.0, 0.0]], dtype=dtype, device=device).expand_as(P)

    return P, V


class Sphere2DSurfaceKernel(FunctionalKernel):
    """
    Functional kernel for a 2D spherical arc parameterized by:
        - signed surface curvature
        - lens diameter
    """

    input_names = ["P", "V", "dfk_in", "ifk_in"]
    param_names = ["C"]
    output_names = ["t", "normals", "valid", "dfk_out", "ifk_out"]

    @staticmethod
    def forward(
        P: Batch2DTensor,
        V: Batch2DTensor,
        dfk: HomMatrix,
        ifk: HomMatrix,
        C: ScalarTensor,
    ):
        # TODO static kernel parameter?
        num_iter = 3

        # Setup the local solver for this surface class
        sag = partial(spherical_sag_2d, C=C)
        local_solver = partial(
            sag_surface_local_raytrace_2d,
            sag_function=sag,
            num_iter=num_iter,
        )

        # Perform raytrace
        t, normals = raytrace(P, V, dfk, ifk, local_solver)

        # for now, all valid
        valid = torch.full_like(t, True)

        return t, normals, valid, dfk + 0, ifk + 0

    @staticmethod
    def example_inputs(
        dtype: torch.dtype, device: torch.device
    ) -> tuple[Batch2DTensor, Batch2DTensor, HomMatrix, HomMatrix]:
        P, V = example_rays_2d(10, dtype, device)
        dfk, ifk = hom_identity_2d(dtype, device)
        return P, V, dfk, ifk

    @staticmethod
    def example_params(
        dtype: torch.dtype, device: torch.device
    ) -> tuple[Float[torch.Tensor, ""]]:
        return (torch.tensor(0.5, dtype=dtype, device=device),)
