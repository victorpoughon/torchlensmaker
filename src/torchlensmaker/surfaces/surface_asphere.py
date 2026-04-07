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
from torchlensmaker.kinematics.homogeneous_geometry import (
    hom_identity_2d,
    hom_identity_3d,
)
from torchlensmaker.surfaces.implicit_solver import implicit_solver_newton
from torchlensmaker.surfaces.sag_geometry import (
    lens_diameter_implicit_domain_2d,
    lens_diameter_implicit_domain_3d,
)
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

from .kernels_utils import example_rays_2d, example_rays_3d
from .sag_functions import (
    aspheric_sag_2d,
    aspheric_sag_3d,
    conical_sag_2d,
    conical_sag_3d,
    sag_sum_2d,
    sag_sum_3d,
    sag_to_implicit_2d_raw,
    sag_to_implicit_3d_raw,
)
from .surface_element import SurfaceElement, SurfaceElementOutput


class AsphereSurfaceKernel(FunctionalKernel):
    """
    Functional kernel for a 2D or 3D asphere surface parameterized by:
        - signed curvature coefficient C
        - conic constant K
        - aspheric coefficients alpha_i
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
        "C": ScalarTensor,
        "K": ScalarTensor,
        "alphas": Float[torch.Tensor, " N"],
        "normalize": Bool[torch.Tensor, ""],
    }

    outputs = {
        "t": BatchTensor,
        "normals": BatchNDTensor,
        "valid": MaskTensor,
        "points_local": BatchNDTensor,
        "points_global": BatchNDTensor,
        "rsm": BatchTensor,
    }

    def __init__(self, dim: int, solver_config: SolverConfig):
        self.dim = dim
        self.solver_config = solver_config

    def apply(
        self,
        P: BatchNDTensor,
        V: BatchNDTensor,
        tf_in: Tf,
        diameter: ScalarTensor,
        C: ScalarTensor,
        K: ScalarTensor,
        alphas: Float[torch.Tensor, " N"],
        normalize: Bool[torch.Tensor, ""],
    ) -> tuple[
        BatchTensor,
        BatchNDTensor,
        MaskTensor,
        BatchNDTensor,
        BatchNDTensor,
        BatchTensor,
    ]:
        if self.dim == 2:
            sag_function = partial(
                sag_sum_2d,
                sags=[
                    partial(conical_sag_2d, C=C, K=K),
                    partial(aspheric_sag_2d, coefficients=alphas),
                ],
            )
        else:
            sag_function = partial(
                sag_sum_3d,
                sags=[
                    partial(conical_sag_3d, C=C, K=K),
                    partial(aspheric_sag_3d, coefficients=alphas),
                ],
            )

        liftf, domainf, implicit_solver = sag_solver_config(
            self.dim, self.solver_config, diameter
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
        ScalarTensor,
        Float[torch.Tensor, " N"],
        Bool[torch.Tensor, ""],
    ]:
        return (
            torch.tensor(10.0, dtype=dtype, device=device),
            torch.tensor(0.5, dtype=dtype, device=device),
            torch.tensor(1.0, dtype=dtype, device=device),
            torch.tensor([0.1, 0.001, 0.002], dtype=dtype, device=device),
            torch.tensor(False, dtype=torch.bool, device=device),
        )


class AsphereOuterExtentSurfaceKernel(FunctionalKernel):
    inputs = {"anchor": ScalarTensor}
    params = {
        "diameter": ScalarTensor,
        "C": ScalarTensor,
        "K": ScalarTensor,
        "alphas": ScalarTensor,
        "normalize": Bool[torch.Tensor, ""],
    }
    outputs = {"extent": ScalarTensor}

    def apply(
        self,
        anchor: ScalarTensor,
        diameter: ScalarTensor,
        C: ScalarTensor,
        K: ScalarTensor,
        alphas: Float[torch.Tensor, " N"],
        normalize: Bool[torch.Tensor, ""],
    ) -> ScalarTensor:
        sag_function = partial(
            sag_sum_2d,
            sags=[
                partial(conical_sag_2d, C=C, K=K),
                partial(aspheric_sag_2d, coefficients=alphas),
            ],
        )
        extent_unnormalized = sag_function(anchor * diameter / 2).val
        extent_normalized = diameter / 2 * sag_function(anchor).val
        extent = torch.where(normalize, extent_normalized, extent_unnormalized)
        return extent

    def example_inputs(
        self, dtype: torch.dtype, device: torch.device
    ) -> tuple[ScalarTensor]:
        return (torch.tensor(1.0, dtype=dtype, device=device),)

    def example_params(
        self, dtype: torch.dtype, device: torch.device
    ) -> tuple[
        ScalarTensor,
        ScalarTensor,
        ScalarTensor,
        Float[torch.Tensor, " N"],
        Bool[torch.Tensor, ""],
    ]:
        return (
            torch.tensor(10.0, dtype=dtype, device=device),
            torch.tensor(0.5, dtype=dtype, device=device),
            torch.tensor(0.5, dtype=dtype, device=device),
            torch.tensor((0.5, 0.0, 0.0), dtype=dtype, device=device),
            torch.tensor(True, dtype=torch.bool, device=device),
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
        self.func2d = AsphereSurfaceKernel(2, self.solver_config)
        self.func3d = AsphereSurfaceKernel(3, self.solver_config)
        self.kernel_outer_extent = AsphereOuterExtentSurfaceKernel()
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

        t, normal, valid, points_local, points_global, rsm = kernel_surface.apply(
            P,
            V,
            tf_surface,
            self.diameter,
            self.C,
            self.K,
            self.alphas,
            self.normalize,
        )

        return SurfaceElementOutput(
            t, normal, valid, points_local, points_global, rsm, tf_surface, tf_next
        )

    def outer_extent(self, anchor: ScalarTensor) -> ScalarTensor:
        return self.kernel_outer_extent.apply(
            anchor, self.diameter, self.C, self.K, self.alphas, self.normalize
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
                        "sag-type": "aspheric",
                        "coefficients": self.alphas.tolist(),
                    },
                ],
            },
        }
