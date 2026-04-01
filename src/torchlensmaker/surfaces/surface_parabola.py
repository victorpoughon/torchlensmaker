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
from typing import Any, Self

import torch
import torch.nn as nn
from jaxtyping import Bool, Float

from torchlensmaker.core.functional_kernel import FunctionalKernel
from torchlensmaker.core.tensor_manip import init_param
from torchlensmaker.kinematics.homogeneous_geometry import (
    hom_identity_2d,
    hom_identity_3d,
)
from torchlensmaker.surfaces.surface_anchor import SurfaceScaleAnchorKernel
from torchlensmaker.types import (
    BatchNDTensor,
    BatchTensor,
    MaskTensor,
    ScalarTensor,
    Tf,
)

from .kernels_utils import example_rays_2d, example_rays_3d
from .sag_functions import (
    parabolic_sag_2d,
    parabolic_sag_3d,
    sag_to_implicit_2d,
    sag_to_implicit_3d,
)
from .sag_surface import sag_surface_2d, sag_surface_3d
from .surface_element import SurfaceElement, SurfaceElementOutput


class ParabolaSurfaceKernel(FunctionalKernel):
    """
    Functional kernel for a 2D or 3D parabolic arc surface parameterized by:
        - signed coefficient A, where x = A*y^2
        - lens diameter

    with support for anchors and scale.
    """

    inputs = {
        "P": BatchNDTensor,
        "V": BatchNDTensor,
        "tf_in": Tf,
    }

    params = {
        "diameter": ScalarTensor,
        "A": ScalarTensor,
        "normalize": Bool[torch.Tensor, ""],
    }

    outputs = {
        "t": BatchTensor,
        "normals": BatchNDTensor,
        "valid": MaskTensor,
        "points_local": BatchNDTensor,
        "points_global": BatchNDTensor,
    }

    def __init__(self, dim: int, num_iter: int, damping: float, tol: float):
        self.dim = dim
        self.num_iter = num_iter
        self.damping = damping
        self.tol = tol

    def apply(
        self,
        P: BatchNDTensor,
        V: BatchNDTensor,
        tf_in: Tf,
        diameter: ScalarTensor,
        A: ScalarTensor,
        normalize: Bool[torch.Tensor, ""],
    ) -> tuple[BatchTensor, BatchNDTensor, MaskTensor, BatchNDTensor, BatchNDTensor]:
        sag_function = parabolic_sag_2d if self.dim == 2 else parabolic_sag_3d
        lift_function = sag_to_implicit_2d if self.dim == 2 else sag_to_implicit_3d
        apply_impl = sag_surface_2d if self.dim == 2 else sag_surface_3d

        return apply_impl(
            partial(sag_function, A=A),
            lift_function,
            self.num_iter,
            self.damping,
            self.tol,
            P,
            V,
            tf_in,
            diameter,
            normalize,
        )

    def example_inputs(
        self, dtype: torch.dtype, device: torch.device
    ) -> tuple[BatchNDTensor, BatchNDTensor, Tf]:
        tf: Tf
        if self.dim == 2:
            P, V = example_rays_2d(10, dtype, device)
            tf = hom_identity_2d(dtype, device)
        else:
            P, V = example_rays_3d(10, dtype, device)
            tf = hom_identity_3d(dtype, device)

        return P, V, tf

    def example_params(
        self, dtype: torch.dtype, device: torch.device
    ) -> tuple[
        ScalarTensor,
        ScalarTensor,
        Bool[torch.Tensor, ""],
    ]:
        return (
            torch.tensor(10.0, dtype=dtype, device=device),
            torch.tensor(0.5, dtype=dtype, device=device),
            torch.tensor(False, dtype=torch.bool, device=device),
        )


class ParabolaOuterExtentSurfaceKernel(FunctionalKernel):
    inputs = {"anchor": ScalarTensor}
    params = {
        "diameter": ScalarTensor,
        "A": ScalarTensor,
        "normalize": Bool[torch.Tensor, ""],
    }
    outputs = {"extent": ScalarTensor}

    def apply(
        self,
        anchor: ScalarTensor,
        diameter: ScalarTensor,
        A: ScalarTensor,
        normalize: Bool[torch.Tensor, ""],
    ) -> ScalarTensor:
        extent_unnormalized = parabolic_sag_2d(anchor * diameter / 2, A)[0]
        extent_normalized = diameter / 2 * parabolic_sag_2d(anchor, A)[0]
        extent = torch.where(normalize, extent_normalized, extent_unnormalized)
        return extent

    def example_inputs(
        self, dtype: torch.dtype, device: torch.device
    ) -> tuple[(ScalarTensor)]:
        return (torch.tensor(1.0, dtype=dtype, device=device),)

    def example_params(
        self, dtype: torch.dtype, device: torch.device
    ) -> tuple[ScalarTensor, ScalarTensor, Bool[torch.Tensor, ""]]:
        return (
            torch.tensor(10.0, dtype=dtype, device=device),
            torch.tensor(0.5, dtype=dtype, device=device),
            torch.tensor(True, dtype=torch.bool, device=device),
        )


class Parabola(SurfaceElement):
    """
    Parabolic surface (2D or 3D) parameterized by lens diameter and parabolic coefficient.

    Represented by a sag function and raytraced by implicit solver
    Support for anchors and scale.
    """

    def __init__(
        self,
        diameter: float | ScalarTensor,
        A: float | ScalarTensor | nn.Parameter,
        *,
        anchors: tuple[float, float] | Float[torch.Tensor, " 2"] = (0.0, 0.0),
        scale: float | ScalarTensor = 1.0,
        trainable: bool = False,
        normalize: bool = False,
        num_iter: int = 12,
        damping: float = 0.95,
        tol: float = 1e-4,
    ):
        super().__init__()
        self.diameter = init_param(self, "diameter", diameter, False)
        self.A = init_param(self, "A", A, trainable)
        self.anchors = init_param(self, "anchors", anchors, False)
        self.scale = init_param(self, "scale", scale, False)
        self.normalize = init_param(
            self, "normalize", normalize, False, default_dtype=torch.bool
        )
        self.func2d = ParabolaSurfaceKernel(2, num_iter, damping, tol)
        self.func3d = ParabolaSurfaceKernel(3, num_iter, damping, tol)
        self.kernel_outer_extent = ParabolaOuterExtentSurfaceKernel()
        self.kernel_anchor2d = SurfaceScaleAnchorKernel(2)
        self.kernel_anchor3d = SurfaceScaleAnchorKernel(3)

    def clone(self, **overrides) -> Self:
        kwargs: dict[str, Any] = dict(
            diameter=self.diameter,
            A=self.A,
            anchors=self.anchors,
            scale=self.scale,
            trainable=self.A.requires_grad,
            normalize=self.normalize,
            num_iter=self.func2d.num_iter,
            damping=self.func2d.damping,
            tol=self.func2d.tol,
        )
        return type(self)(**kwargs | overrides)

    def reverse(self) -> Self:
        return self.clone(anchors=self.anchors.flip(0))

    def forward(self, P: BatchTensor, V: BatchTensor, tf: Tf) -> SurfaceElementOutput:
        kernel_surface = self.func2d if tf.pdim() == 2 else self.func3d
        kernel_anchor = self.kernel_anchor2d if tf.pdim() == 2 else self.kernel_anchor3d

        extent0 = self.outer_extent(self.anchors[0])
        extent1 = self.outer_extent(self.anchors[1])

        tf_surface, tf_next = kernel_anchor.apply(extent0, extent1, self.scale, tf)

        t, normal, valid, points_local, points_global = kernel_surface.apply(
            P,
            V,
            tf_surface,
            self.diameter,
            self.A,
            self.normalize,
        )

        return SurfaceElementOutput(
            t, normal, valid, points_local, points_global, tf_surface, tf_next
        )

    def outer_extent(self, anchor: ScalarTensor) -> ScalarTensor:
        return self.kernel_outer_extent.apply(
            anchor, self.diameter, self.A, self.normalize
        )

    def render(self) -> Any:
        return {
            "type": "surface-sag",
            "diameter": self.diameter.item(),
            "sag-function": {
                "sag-type": "parabolic",
                "A": self.A.item(),
                "normalize": self.normalize.item(),
            },
        }
