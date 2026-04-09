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
from torchlensmaker.surfaces.implicit_solver import implicit_solver_newton
from torchlensmaker.surfaces.sag_geometry import (
    lens_diameter_implicit_domain_2d,
    lens_diameter_implicit_domain_3d,
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
    conical_sag_2d,
    conical_sag_3d,
)
from .sag_surface import SolverConfig, sag_solver_config, sag_surface_raytrace
from .surface_element import SurfaceElement, SurfaceElementOutput


class ConicSurfaceKernel(FunctionalKernel):
    """
    Functional kernel for a 2D or 3D conical arc surface parameterized by:
        - signed curvature coefficient C
        - conic constant K
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
        normalize: Bool[torch.Tensor, ""],
    ) -> tuple[
        BatchTensor,
        BatchNDTensor,
        MaskTensor,
        BatchNDTensor,
        BatchNDTensor,
        BatchTensor,
    ]:
        sag_function = conical_sag_2d if self.dim == 2 else conical_sag_3d
        liftf, domainf, implicit_solver = sag_solver_config(
            self.dim, self.solver_config, diameter
        )

        return sag_surface_raytrace(
            partial(sag_function, C=C, K=K),
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
        Bool[torch.Tensor, ""],
    ]:
        return (
            torch.tensor(10.0, dtype=dtype, device=device),
            torch.tensor(0.5, dtype=dtype, device=device),
            torch.tensor(1.0, dtype=dtype, device=device),
            torch.tensor(False, dtype=torch.bool, device=device),
        )


class ConicOuterExtentSurfaceKernel(FunctionalKernel):
    inputs = {"anchor": ScalarTensor}
    params = {
        "diameter": ScalarTensor,
        "C": ScalarTensor,
        "K": ScalarTensor,
        "normalize": Bool[torch.Tensor, ""],
    }
    outputs = {"extent": ScalarTensor}

    def apply(
        self,
        anchor: ScalarTensor,
        diameter: ScalarTensor,
        C: ScalarTensor,
        K: ScalarTensor,
        normalize: Bool[torch.Tensor, ""],
    ) -> ScalarTensor:
        extent_unnormalized = conical_sag_2d(anchor * diameter / 2, C, K, order=0).val
        extent_normalized = diameter / 2 * conical_sag_2d(anchor, C, K, order=0).val
        extent = torch.where(normalize, extent_normalized, extent_unnormalized)
        return extent

    def example_inputs(
        self, dtype: torch.dtype, device: torch.device
    ) -> tuple[(ScalarTensor)]:
        return (torch.tensor(1.0, dtype=dtype, device=device),)

    def example_params(
        self, dtype: torch.dtype, device: torch.device
    ) -> tuple[ScalarTensor, ScalarTensor, ScalarTensor, Bool[torch.Tensor, ""]]:
        return (
            torch.tensor(10.0, dtype=dtype, device=device),
            torch.tensor(0.5, dtype=dtype, device=device),
            torch.tensor(0.5, dtype=dtype, device=device),
            torch.tensor(True, dtype=torch.bool, device=device),
        )


class Conic(SurfaceElement):
    """
    Conic surface (2D or 3D) parameterized by:
    - lens diameter
    - signed curvature C
    - conic contant K

    Represented by a sag function and raytraced by implicit solver
    Support for anchors and scale.
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
        self.anchors = init_param(self, "anchors", anchors, False)
        self.scale = init_param(self, "scale", scale, False)
        self.normalize = init_param(
            self, "normalize", normalize, False, default_dtype=torch.bool
        )
        self.func2d = ConicSurfaceKernel(2, self.solver_config)
        self.func3d = ConicSurfaceKernel(3, self.solver_config)
        self.kernel_outer_extent = ConicOuterExtentSurfaceKernel()
        self.kernel_anchor2d = SurfaceScaleAnchorKernel(2)
        self.kernel_anchor3d = SurfaceScaleAnchorKernel(3)

    def clone(self, **overrides: Any) -> Self:
        kwargs: dict[str, Any] = dict(
            diameter=self.diameter,
            C=self.C,
            K=self.K,
            anchors=self.anchors,
            scale=self.scale,
            trainable=self.C.requires_grad,
            normalize=self.normalize,
            solver_config=self.solver_config,
        )
        return type(self)(**kwargs | overrides)

    def __repr__(self) -> str:
        return f"{self._get_name()}(diameter={self.diameter.item()}, C={self.C.item()}, K={self.K.item()})"

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
            self.normalize,
        )

        return SurfaceElementOutput(
            t, normal, valid, points_local, points_global, rsm, tf_surface, tf_next
        )

    def outer_extent(self, anchor: ScalarTensor) -> ScalarTensor:
        return self.kernel_outer_extent.apply(
            anchor, self.diameter, self.C, self.K, self.normalize
        )

    def render(self) -> Any:
        return {
            "type": "surface-sag",
            "diameter": self.diameter.item(),
            "sag-function": {
                "sag-type": "conical",
                "C": self.C.item(),
                "K": self.K.item(),
                "normalize": self.normalize.item(),
            },
        }
