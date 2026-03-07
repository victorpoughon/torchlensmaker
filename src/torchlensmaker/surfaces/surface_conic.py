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
from jaxtyping import Float, Bool
import torch
import torch.nn as nn

from torchlensmaker.types import (
    ScalarTensor,
    BatchTensor,
    BatchNDTensor,
    MaskTensor,
    Tf,
    Direction,
)

from .surface_element import SurfaceElement

from torchlensmaker.kinematics.homogeneous_geometry import (
    hom_identity_2d,
    hom_identity_3d,
)

from torchlensmaker.core.functional_kernel import FunctionalKernel
from torchlensmaker.core.tensor_manip import init_param
from torchlensmaker.surfaces.surface_anchor import SurfaceScaleAnchorKernel
from .sag_functions import conical_sag_2d, conical_sag_3d

from .kernels_utils import example_rays_2d, example_rays_3d
from .sag_surface import sag_surface_2d, sag_surface_3d


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
        C: ScalarTensor,
        K: ScalarTensor,
        normalize: Bool[torch.Tensor, ""],
    ) -> tuple[BatchTensor, BatchNDTensor, MaskTensor]:
        sag_function = conical_sag_2d if self.dim == 2 else conical_sag_3d
        apply_impl = sag_surface_2d if self.dim == 2 else sag_surface_3d

        return apply_impl(
            partial(sag_function, C=C, K=K),
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
        ScalarTensor,
        Float[torch.Tensor, " 2"],
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
    inputs = {"r": ScalarTensor, "C": ScalarTensor, "K": ScalarTensor}
    params = {}
    outputs = {"extent": ScalarTensor}

    def apply(self, r: ScalarTensor, C: ScalarTensor, K: ScalarTensor) -> ScalarTensor:
        extent, _ = conical_sag_2d(r, C, K)
        return extent

    def example_inputs(
        self, dtype: torch.dtype, device: torch.device
    ) -> tuple[ScalarTensor, ScalarTensor]:
        return (
            torch.tensor(10.0, dtype=dtype, device=device),
            torch.tensor(0.5, dtype=dtype, device=device),
            torch.tensor(0.5, dtype=dtype, device=device),
        )

    def example_params(self, dtype: torch.dtype, device: torch.device) -> tuple[()]:
        return tuple()


class Conic(SurfaceElement):
    """
    Conic surface (2D or 3D) parameterized by:
    - lens diameter
    - signed curvature C
    - conic contant K

    Represented by a sag function and raytraced by implicit solver
    Support for anchors and scale.
    """

    def __init__(
        self,
        diameter: float | ScalarTensor,
        C: float | ScalarTensor | nn.Parameter,
        K: float | ScalarTensor | nn.Parameter,
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
        self.K = init_param(self, "K", K, trainable)
        self.anchors = init_param(self, "anchors", anchors, False)
        self.scale = init_param(self, "scale", scale, False)
        self.normalize = init_param(
            self, "normalize", normalize, False, default_dtype=torch.bool
        )
        self.func2d = ConicSurfaceKernel(2, num_iter, damping, tol)
        self.func3d = ConicSurfaceKernel(3, num_iter, damping, tol)
        self.kernel_outer_extent = ConicOuterExtentSurfaceKernel()
        self.kernel_anchor2d = SurfaceScaleAnchorKernel(2)
        self.kernel_anchor3d = SurfaceScaleAnchorKernel(3)

    def clone(self, **overrides: Any) -> Self:
        kwargs = dict(
            diameter=self.diameter,
            C=self.C,
            K=self.K,
            anchors=self.anchors,
            scale=self.scale,
            trainable=self.C.requires_grad,
            normalize=self.normalize,
            num_iter=self.func2d.num_iter,
            damping=self.func2d.damping,
            tol=self.func2d.tol,
        )
        return type(self)(**kwargs | overrides)

    def forward(
        self, P: BatchTensor, V: BatchTensor, tf: Tf, direction: Direction
    ) -> tuple[BatchTensor, BatchNDTensor, MaskTensor, Tf, Tf]:
        kernel_surface = self.func2d if tf.pdim() == 2 else self.func3d
        kernel_anchor = self.kernel_anchor2d if tf.pdim() == 2 else self.kernel_anchor3d

        # Retrograde direction just needs to swap anchors
        anchors = (
            self.anchors.unbind(-1)
            if direction.is_prograde()
            else self.anchors.flip(0).unbind(-1)
        )

        extent0 = self.kernel_outer_extent.apply(
            anchors[0] * self.diameter / 2, self.C, self.K
        )
        extent1 = self.kernel_outer_extent.apply(
            anchors[1] * self.diameter / 2, self.C, self.K
        )

        tf_surface, tf_next = kernel_anchor.apply(extent0, extent1, self.scale, tf)

        t, normal, valid = kernel_surface.apply(
            P,
            V,
            tf,
            self.diameter,
            self.C,
            self.K,
            self.normalize,
        )

        return t, normal, valid, tf_surface, tf_next

    def outer_extent(self, r: ScalarTensor) -> ScalarTensor | None:
        return self.kernel_outer_extent.apply(r, self.C, self.K)

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
