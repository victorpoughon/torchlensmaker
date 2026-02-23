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
from jaxtyping import Float, Bool
import torch
import torch.nn as nn

from torchlensmaker.types import (
    ScalarTensor,
    BatchTensor,
    Batch2DTensor,
    Batch3DTensor,
    BatchNDTensor,
    MaskTensor,
    HomMatrix,
    Tf2D,
    Tf3D,
    Tf,
)

from torchlensmaker.kinematics.homogeneous_geometry import (
    hom_identity_2d,
    hom_identity_3d,
)

from torchlensmaker.core.functional_kernel import FunctionalKernel
from torchlensmaker.core.tensor_manip import init_param

from .sag_functions import (
    spherical_sag_2d,
    sag_to_implicit_2d,
    spherical_sag_3d,
    sag_to_implicit_3d,
)
from .implicit_solver import implicit_surface_local_raytrace

from .raytrace import raytrace
from .sag_geometry import (
    lens_diameter_implicit_domain_2d,
    lens_diameter_implicit_domain_3d,
    anchor_transforms_2d,
    anchor_transforms_3d,
)
from .kernels_utils import example_rays_2d, example_rays_3d


class SphereByCurvature2DSurfaceKernel(FunctionalKernel):
    """
    Functional kernel for a 2D spherical arc parameterized by:
        - signed surface curvature
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
        "C": ScalarTensor,
        "anchors": Float[torch.Tensor, " 2"],
        "scale": ScalarTensor,
        "normalize": Bool[torch.Tensor, ""],
    }

    outputs = {
        "t": BatchTensor,
        "normals": Batch2DTensor,
        "valid": MaskTensor,
        "surface_tf": Tf2D,
        "next_tf": Tf2D,
    }

    def __init__(self, num_iter: int, damping: float, tol: float):
        self.num_iter = num_iter
        self.damping = damping
        self.tol = tol

    def apply(
        self,
        P: Batch2DTensor,
        V: Batch2DTensor,
        tf_in: Tf2D,
        diameter: ScalarTensor,
        C: ScalarTensor,
        anchors: Float[torch.Tensor, " 2"],
        scale: ScalarTensor,
        normalize: Bool[torch.Tensor, ""],
    ) -> tuple[BatchTensor, Batch2DTensor, MaskTensor, Tf2D, Tf2D]:
        # Setup the local solver for this surface class
        sag = partial(spherical_sag_2d, C=C)
        one = torch.ones((), dtype=P.dtype, device=P.device)
        tau = torch.where(normalize, diameter / 2, one)
        implicit_function = sag_to_implicit_2d(sag, tau=tau)
        domain_function = partial(
            lens_diameter_implicit_domain_2d, diameter=diameter, tol=self.tol
        )
        local_solver = partial(
            implicit_surface_local_raytrace,
            implicit_function=implicit_function,
            domain_function=domain_function,
            num_iter=self.num_iter,
            damping=self.damping,
        )

        # Compute anchor transforms from anchors and scale
        extent0, _ = sag(anchors[0] * diameter / 2)
        extent1, _ = sag(anchors[1] * diameter / 2)
        tf_surface, tf_next = anchor_transforms_2d(extent0, extent1, scale, tf_in)

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
            torch.tensor(False, dtype=torch.bool, device=device),
        )


class SphereByCurvature3DSurfaceKernel(FunctionalKernel):
    """
    Functional kernel for a 3D spherical cap parameterized by:
        - signed surface curvature
        - lens diameter

    with support for anchors and scale.
    """

    inputs = {
        "P": Batch3DTensor,
        "V": Batch3DTensor,
        "tf_in": Tf3D,
    }

    params = {
        "diameter": ScalarTensor,
        "C": ScalarTensor,
        "anchors": Float[torch.Tensor, " 2"],
        "scale": ScalarTensor,
        "normalize": Bool[torch.Tensor, ""],
    }

    outputs = {
        "t": BatchTensor,
        "normals": Batch3DTensor,
        "valid": MaskTensor,
        "surface_tf": Tf3D,
        "next_tf": Tf3D,
    }

    def __init__(self, num_iter: int, damping: float, tol: float):
        self.num_iter = num_iter
        self.damping = damping
        self.tol = tol

    def apply(
        self,
        P: Batch3DTensor,
        V: Batch3DTensor,
        tf_in: Tf3D,
        diameter: ScalarTensor,
        C: ScalarTensor,
        anchors: Float[torch.Tensor, " 2"],
        scale: ScalarTensor,
        normalize: Bool[torch.Tensor, ""],
    ) -> tuple[BatchTensor, Batch3DTensor, MaskTensor, Tf3D, Tf3D]:
        # Setup the local solver for this surface class
        sag = partial(spherical_sag_3d, C=C)
        one = torch.ones((), dtype=P.dtype, device=P.device)
        tau = torch.where(normalize, diameter / 2, one)
        implicit_function = sag_to_implicit_3d(sag, tau=tau)
        domain_function = partial(
            lens_diameter_implicit_domain_3d, diameter=diameter, tol=self.tol
        )
        local_solver = partial(
            implicit_surface_local_raytrace,
            implicit_function=implicit_function,
            domain_function=domain_function,
            num_iter=self.num_iter,
            damping=self.damping,
        )

        # Compute anchor transforms from anchors and scale
        # In 3D, take Y=tau, Z=0 as the extent point, but this is arbitrary
        extent0, _ = sag(anchors[0] * diameter / 2, torch.zeros_like(anchors[0]))
        extent1, _ = sag(anchors[1] * diameter / 2, torch.zeros_like(anchors[0]))
        tf_surface, tf_next = anchor_transforms_3d(extent0, extent1, scale, tf_in)

        # Perform raytrace
        t, normals, valid = raytrace(P, V, tf_surface, local_solver)

        return t, normals, valid, tf_surface, tf_next

    def example_inputs(
        self, dtype: torch.dtype, device: torch.device
    ) -> tuple[Batch3DTensor, Batch3DTensor, Tf3D]:
        P, V = example_rays_3d(10, dtype, device)
        tf = hom_identity_3d(dtype, device)
        return P, V, tf

    def example_params(
        self, dtype: torch.dtype, device: torch.device
    ) -> tuple[ScalarTensor, ScalarTensor, Float[torch.Tensor, " 2"], ScalarTensor]:
        return (
            torch.tensor(10.0, dtype=dtype, device=device),
            torch.tensor(0.5, dtype=dtype, device=device),
            torch.tensor((0.0, 0.0), dtype=dtype, device=device),
            torch.tensor(-1.0, dtype=dtype, device=device),
            torch.tensor(False, dtype=torch.bool, device=device),
        )


class SphereByCurvature(nn.Module):
    """
    Spherical surface (2D or 3D) parameterized by lens diameter and curvature.

    Represented by a sag function and raytraced by implicit solver
    Support for anchors and scale.
    """

    def __init__(
        self,
        diameter: float | ScalarTensor,
        C: float | ScalarTensor | nn.Parameter,
        *,
        anchors: tuple[float, float] | Float[torch.Tensor, " 2"] = (0.0, 0.0),
        scale: float | ScalarTensor = 1.0,
        trainable: bool = True,
        normalize: bool = False,
        num_iter: int = 6,
        damping: float = 0.95,
        tol: float = 1e-6,
    ):
        super().__init__()
        self.diameter = init_param(self, "diameter", diameter, False)
        self.C = init_param(self, "C", C, trainable)
        self.anchors = init_param(self, "anchors", anchors, False)
        self.scale = init_param(self, "scale", scale, False)
        self.normalize = init_param(
            self, "normalize", normalize, False, default_dtype=torch.bool
        )
        self.func2d = SphereByCurvature2DSurfaceKernel(num_iter, damping, tol)
        self.func3d = SphereByCurvature3DSurfaceKernel(num_iter, damping, tol)

    def forward(
        self, P: BatchTensor, V: BatchTensor, tf: Tf
    ) -> tuple[BatchTensor, BatchNDTensor, MaskTensor, Tf, Tf]:
        if P.shape[-1] == 2:
            return self.func2d.apply(
                P,
                V,
                tf,
                self.diameter,
                self.C,
                self.anchors,
                self.scale,
                self.normalize,
            )
        else:
            return self.func3d.apply(
                P,
                V,
                tf,
                self.diameter,
                self.C,
                self.anchors,
                self.scale,
                self.normalize,
            )

    def render(self) -> Any:
        return {
            "type": "surface-sag",
            "diameter": self.diameter.item(),
            "sag-function": {
                "sag-type": "spherical",
                "C": self.C.item(),
                "normalize": self.normalize.item(),
            },
        }
