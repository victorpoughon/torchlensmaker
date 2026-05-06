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
from torchlensmaker.surfaces.surface_sag import SagSumSurfaceKernel, SagSurfaceKernel
from torchlensmaker.types import (
    BatchTensor,
    ScalarTensor,
    Tf,
)

from .surface_element import SurfaceElement, SurfaceElementOutput


class XYPolynomial(SurfaceElement):
    """
    XYPolynomial 3D surface parameterized by:
    - lens diameter
    - signed curvature C
    - conic contant K
    - xypolynomial coefficients

    Represented by a sag function and raytraced by implicit solver
    Support for scale.
    """

    default_config = SolverConfig(
        implicit_solver="newton",
        num_iter=6,
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
        coefficients: Sequence[Sequence[float]]
        | Float[torch.Tensor, "P Q"]
        | nn.Parameter,
        *,
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
        self.coefficients = init_param(self, "coefficients", coefficients, trainable)
        self.scale = init_param(self, "scale", scale, False)
        self.normalize = init_param(
            self, "normalize", normalize, False, default_dtype=torch.bool
        )
        self.func3d = SagSumSurfaceKernel(
            3, ti.conical_sag_3d, ti.xypolynomial_sag_3d, self.solver_config
        )
        self.kernel_anchor3d = SurfaceScaleAnchorKernel(3)

    def clone(self, **overrides: Any) -> Self:
        kwargs: dict[str, Any] = dict(
            diameter=self.diameter,
            C=self.C,
            K=self.K,
            coefficients=self.coefficients,
            scale=self.scale,
            trainable=self.C.requires_grad,
            normalize=self.normalize,
            solver_config=self.solver_config,
        )
        return type(self)(**kwargs | overrides)

    def reverse(self) -> Self:
        return self.clone()

    def forward(self, P: BatchTensor, V: BatchTensor, tf: Tf) -> SurfaceElementOutput:
        # TODO raise nice error if 2D
        kernel_surface = self.func3d
        kernel_anchor = self.kernel_anchor3d

        zero = torch.zeros((), dtype=P.dtype, device=P.device)
        tf_surface, tf_next = kernel_anchor.apply(zero, zero, self.scale, tf)

        params1, params2 = torch.stack([self.C, self.K]), self.coefficients

        t, normal, valid, points_local, points_global, rsm = kernel_surface.apply(
            P,
            V,
            tf_surface,
            self.diameter,
            self.normalize,
            params1,
            params2,
        )

        return SurfaceElementOutput(
            t, normal, valid, points_local, points_global, rsm, tf_surface, tf_next
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
                        "sag-type": "xypolynomial",
                        "coefficients": self.coefficients.tolist(),
                    },
                ],
            },
            matrix=matrix.tolist(),
        )
