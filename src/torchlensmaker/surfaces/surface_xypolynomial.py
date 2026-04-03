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

import torch
import torch.nn as nn
from jaxtyping import Bool, Float

from torchlensmaker.core.functional_kernel import FunctionalKernel
from torchlensmaker.core.tensor_manip import init_param
from torchlensmaker.kinematics.homogeneous_geometry import hom_identity_3d
from torchlensmaker.surfaces.implicit_solver import implicit_solver_newton
from torchlensmaker.surfaces.sag_geometry import lens_diameter_implicit_domain_3d
from torchlensmaker.surfaces.sag_surface import (
    SolverConfig,
    sag_solver_config,
    sag_surface_raytrace,
)
from torchlensmaker.surfaces.surface_anchor import SurfaceScaleAnchorKernel
from torchlensmaker.types import (
    BatchNDTensor,
    BatchTensor,
    MaskTensor,
    ScalarTensor,
    Tf,
)

from .kernels_utils import example_rays_3d
from .sag_functions import (
    conical_sag_3d,
    sag_sum_3d,
    sag_to_implicit_3d_raw,
    xypolynomial_sag_3d,
)
from .surface_element import SurfaceElement, SurfaceElementOutput


class XYPolynomialSurfaceKernel(FunctionalKernel):
    """
    Functional kernel for a 3D freeform surface made of a conical base and
    XYPolynomial coefficients. It is parameterized by:
        - signed curvature C
        - conic constant K
        - XY polynomial coefficients
        - lens diameter

    with support for scale.
    """

    inputs = {
        "P": BatchNDTensor,
        "V": BatchNDTensor,
        "tf_in": Tf,
    }

    params = {
        "diameter": ScalarTensor,
        "C": ScalarTensor,
        "K": ScalarTensor,
        "coefficients": Float[torch.Tensor, " P Q"],
        "normalize": Bool[torch.Tensor, ""],
    }

    outputs = {
        "t": BatchTensor,
        "normals": BatchNDTensor,
        "valid": MaskTensor,
        "points_local": BatchNDTensor,
        "points_global": BatchNDTensor,
    }

    def __init__(self, solver_config: SolverConfig):
        self.solver_config = solver_config

    def apply(
        self,
        P: BatchNDTensor,
        V: BatchNDTensor,
        tf_in: Tf,
        diameter: ScalarTensor,
        C: ScalarTensor,
        K: ScalarTensor,
        coefficients: Float[torch.Tensor, " P Q"],
        normalize: Bool[torch.Tensor, ""],
    ) -> tuple[BatchTensor, BatchNDTensor, MaskTensor, BatchNDTensor, BatchNDTensor]:
        # XYPolynomial is 3D only because it's freeform
        sag_function = partial(
            sag_sum_3d,
            sags=[
                partial(conical_sag_3d, C=C, K=K),
                partial(xypolynomial_sag_3d, coefficients=coefficients),
            ],
        )

        liftf, domainf, implicit_solver = sag_solver_config(
            3, self.solver_config, diameter
        )

        return sag_surface_raytrace(
            sag_function,
            liftf,
            domainf,
            implicit_solver,
            P,
            V,
            tf_in,
            diameter,
            normalize,
        )

    def example_inputs(
        self, dtype: torch.dtype, device: torch.device
    ) -> tuple[BatchNDTensor, BatchNDTensor, Tf]:
        P, V = example_rays_3d(10, dtype, device)
        tf = hom_identity_3d(dtype, device)
        return P, V, tf

    def example_params(
        self, dtype: torch.dtype, device: torch.device
    ) -> tuple[
        ScalarTensor,
        ScalarTensor,
        ScalarTensor,
        Float[torch.Tensor, "P Q"],
        Bool[torch.Tensor, ""],
    ]:
        return (
            torch.tensor(10.0, dtype=dtype, device=device),
            torch.tensor(0.5, dtype=dtype, device=device),
            torch.tensor(1.0, dtype=dtype, device=device),
            torch.tensor(
                [[0.1, 0.001, 0.002], [0.1, 0.0, 0.0]], dtype=dtype, device=device
            ),
            torch.tensor(False, dtype=torch.bool, device=device),
        )


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
        self.func3d = XYPolynomialSurfaceKernel(self.solver_config)
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

        t, normal, valid, points_local, points_global = kernel_surface.apply(
            P,
            V,
            tf_surface,
            self.diameter,
            self.C,
            self.K,
            self.coefficients,
            self.normalize,
        )

        return SurfaceElementOutput(
            t, normal, valid, points_local, points_global, tf_surface, tf_next
        )

    def render(self) -> Any:
        return {
            "type": "surface-sag",
            "diameter": self.diameter.item(),
            "sag-function": {
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
        }
