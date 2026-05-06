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
from typing import Any, Self, Sequence

import tlmviewer as tlmv
import torch
import torch.nn as nn
import torchimplicit as ti
from jaxtyping import Float

from torchlensmaker.core.tensor_manip import init_param
from torchlensmaker.surfaces.sag_surface import (
    SolverConfig,
)
from torchlensmaker.surfaces.surface_anchor import SurfaceScaleAnchorKernel
from torchlensmaker.types import (
    BatchTensor,
    ScalarTensor,
    Tf,
)

from .surface_element import SurfaceElement, SurfaceElementOutput
from .surface_sag import SagOuterExtentSurfaceKernel, SagSurfaceKernel


def asphere_sag_2d_eval(points: torch.Tensor, params: torch.Tensor, *, order: int):
    CK = params[0:2]
    alphas = params[2:]

    return ti.sag_sum_2d(
        points,
        sags=[
            partial(ti.conical_sag_2d, params=CK),
            partial(ti.aspheric_sag_2d, params=alphas),
        ],
        order=order,
    )


asphere_sag_2d = ti.SagFunction(
    name="asphere_2d",
    dim=2,
    func=asphere_sag_2d_eval,
    example_params=ti.example_vector([1 / 50, 0.1, 1.0, 0.1, 0.001]),
)


def asphere_sag_3d_eval(points: torch.Tensor, params: torch.Tensor, *, order: int):
    CK = params[0:2]
    alphas = params[2:]

    return ti.sag_sum_3d(
        points,
        sags=[
            partial(ti.conical_sag_3d, params=CK),
            partial(ti.aspheric_sag_3d, params=alphas),
        ],
        order=order,
    )


asphere_sag_3d = ti.SagFunction(
    name="asphere_3d",
    dim=3,
    func=asphere_sag_3d_eval,
    example_params=ti.example_vector([1 / 50, 0.1, 1.0, 0.1, 0.001]),
)


class Asphere(SurfaceElement):
    """
    Asphere surface (2D or 3D) parameterized by:
    - lens diameter
    - signed curvature C
    - conic contant K
    - aspheric coefficients alpha_i

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
        K: float | ScalarTensor | nn.Parameter,
        alphas: Sequence[float] | Float[torch.Tensor, " N"] | nn.Parameter,
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
        self.K = init_param(self, "K", K, trainable)
        self.alphas = init_param(self, "alphas", alphas, trainable)
        self.anchors = init_param(self, "anchors", anchors, False)
        self.scale = init_param(self, "scale", scale, False)
        self.normalize = init_param(
            self, "normalize", normalize, False, default_dtype=torch.bool
        )
        self.func2d = SagSurfaceKernel(2, asphere_sag_2d, self.solver_config)
        self.func3d = SagSurfaceKernel(3, asphere_sag_3d, self.solver_config)
        self.kernel_outer_extent = SagOuterExtentSurfaceKernel(asphere_sag_2d)
        self.kernel_anchor2d = SurfaceScaleAnchorKernel(2)
        self.kernel_anchor3d = SurfaceScaleAnchorKernel(3)

    def clone(self, **overrides: Any) -> Self:
        kwargs: dict[str, Any] = dict(
            diameter=self.diameter,
            C=self.C,
            K=self.K,
            alphas=self.alphas,
            anchors=self.anchors,
            scale=self.scale,
            trainable=self.C.requires_grad,
            normalize=self.normalize,
            solver_config=self.solver_config,
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

        params = torch.cat([self.C.unsqueeze(0), self.K.unsqueeze(0), self.alphas])

        t, normal, valid, points_local, points_global, rsm = kernel_surface.apply(
            P,
            V,
            tf_surface,
            self.diameter,
            self.normalize,
            params,
        )

        return SurfaceElementOutput(
            t, normal, valid, points_local, points_global, rsm, tf_surface, tf_next
        )

    def outer_extent(self, anchor: ScalarTensor) -> ScalarTensor:
        params = torch.cat([self.C.unsqueeze(0), self.K.unsqueeze(0), self.alphas])
        return self.kernel_outer_extent.apply(
            anchor,
            self.diameter,
            self.normalize,
            params,
        )

    def render(self, matrix: torch.Tensor) -> tlmv.SurfaceSag:
        return tlmv.SurfaceSag(
            diameter=self.diameter.item(),
            sag_function={
                "sag-type": "sum",
                "terms": [
                    {
                        "sag-type": "conical",
                        "C": self.C.item(),
                        "K": self.K.item(),
                    },
                    {
                        "sag-type": "aspheric",
                        "coefficients": self.alphas.tolist(),
                    },
                ],
            },
            matrix=matrix.tolist(),
        )
