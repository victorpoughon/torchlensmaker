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

import tlmviewer as tlmv
import torch
import torch.nn as nn
import torchimplicit as ti
from jaxtyping import Float

from torchlensmaker.core.tensor_manip import init_param
from torchlensmaker.surfaces.sag_surface import SolverConfig
from torchlensmaker.surfaces.surface_anchor import SurfaceScaleAnchorKernel
from torchlensmaker.types import (
    BatchTensor,
    ScalarTensor,
    Tf,
)

from .surface_element import SurfaceElement, SurfaceElementOutput
from .surface_sag import SagOuterExtentSurfaceKernel, SagSurfaceKernel


class SphereByCurvature(SurfaceElement):
    """
    Spherical surface (2D or 3D) parameterized by lens diameter and curvature.

    Represented by a sag function and raytraced by implicit solver
    Support for anchors and scale.
    """

    default_config = SolverConfig(
        implicit_solver="newton",
        num_iter=8,
        damping=0.95,
        tol=1e-4,
        lift_function="raw",
        init="closest",
        clamp_positive=True,
    )

    def __init__(
        self,
        diameter: float | ScalarTensor,
        C: float | ScalarTensor | nn.Parameter,
        *,
        anchors: tuple[float, float] | Float[torch.Tensor, " 2"] = (0.0, 0.0),
        scale: float | ScalarTensor = 1.0,
        trainable: bool = False,
        normalize: bool = False,
        solver_config: dict[str, Any] = {},
    ):
        super().__init__()
        self.solver_config = SolverConfig(**self.default_config | solver_config)
        self.diameter = init_param(self, "diameter", diameter, False)
        self.C = init_param(self, "C", C, trainable)
        self.anchors = init_param(self, "anchors", anchors, False)
        self.scale = init_param(self, "scale", scale, False)
        self.normalize = init_param(
            self, "normalize", normalize, False, default_dtype=torch.bool
        )
        self.func2d = SagSurfaceKernel(2, ti.spherical_sag_2d, self.solver_config)
        self.func3d = SagSurfaceKernel(3, ti.spherical_sag_3d, self.solver_config)
        self.kernel_outer_extent = SagOuterExtentSurfaceKernel(ti.spherical_sag_2d)
        self.kernel_anchor2d = SurfaceScaleAnchorKernel(2)
        self.kernel_anchor3d = SurfaceScaleAnchorKernel(3)

    def clone(self, **overrides) -> Self:
        kwargs: dict[str, Any] = dict(
            diameter=self.diameter,
            C=self.C,
            anchors=self.anchors,
            scale=self.scale,
            trainable=self.C.requires_grad,
            normalize=self.normalize,
            solver_config=self.solver_config,
        )
        return type(self)(**kwargs | overrides)

    def __repr__(self) -> str:
        return f"{self._get_name()}(diameter={self.diameter.item()}, C={self.C.item()})"

    def reverse(self) -> Self:
        return self.clone(anchors=self.anchors.flip(0))

    def forward(self, P: BatchTensor, V: BatchTensor, tf: Tf) -> SurfaceElementOutput:
        kernel_surface = self.func2d if tf.pdim() == 2 else self.func3d
        kernel_anchor = self.kernel_anchor2d if tf.pdim() == 2 else self.kernel_anchor3d

        extent0 = self.outer_extent(self.anchors[0])
        extent1 = self.outer_extent(self.anchors[1])

        tf_surface, tf_next = kernel_anchor.apply(extent0, extent1, self.scale, tf)

        t, normal, valid, points_local, points_global, rsm = kernel_surface.apply(
            P,
            V,
            tf_surface,
            self.diameter,
            self.normalize,
            self.C.unsqueeze(0),
        )

        return SurfaceElementOutput(
            t, normal, valid, points_local, points_global, rsm, tf_surface, tf_next
        )

    def outer_extent(self, anchor: ScalarTensor) -> ScalarTensor:
        return self.kernel_outer_extent.apply(
            anchor, self.diameter, self.normalize, self.C.unsqueeze(0)
        )

    def render(self, matrix: torch.Tensor) -> tlmv.SurfaceSag:
        return tlmv.SurfaceSag(
            diameter=self.diameter.item(),
            sag_function={
                "sag-type": "spherical",
                "C": self.C.item(),
                "normalize": self.normalize.item(),
            },
            matrix=matrix.tolist(),
        )
