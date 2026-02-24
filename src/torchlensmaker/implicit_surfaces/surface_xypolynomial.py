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
from typing import Any, Sequence
from jaxtyping import Float, Bool
import torch
import torch.nn as nn

from torchlensmaker.types import (
    ScalarTensor,
    BatchTensor,
    BatchNDTensor,
    MaskTensor,
    Tf,
)

from torchlensmaker.kinematics.homogeneous_geometry import hom_identity_3d
from torchlensmaker.core.functional_kernel import FunctionalKernel
from torchlensmaker.core.tensor_manip import init_param

from .sag_functions import (
    conical_sag_3d,
    xypolynomial_sag_3d,
    sag_sum_3d,
)

from .kernels_utils import example_rays_3d
from .sag_surface import sag_surface_3d


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
        "scale": ScalarTensor,
        "normalize": Bool[torch.Tensor, ""],
    }

    outputs = {
        "t": BatchTensor,
        "normals": BatchNDTensor,
        "valid": MaskTensor,
        "surface_tf": Tf,
        "next_tf": Tf,
    }

    def __init__(self, num_iter: int, damping: float, tol: float):
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
        coefficients: Float[torch.Tensor, " P Q"],
        scale: ScalarTensor,
        normalize: Bool[torch.Tensor, ""],
    ) -> tuple[BatchTensor, BatchNDTensor, MaskTensor, Tf, Tf]:
        # XYPolynomial is 3D only because it's freeform
        sag_function = partial(
            sag_sum_3d,
            sags=[
                partial(conical_sag_3d, C=C, K=K),
                partial(xypolynomial_sag_3d, coefficients=coefficients),
            ],
        )
        
        # XYPolynomial is 3D freeform so it doesn't support anchors
        anchors = torch.zeros((2,), dtype=P.dtype, device=P.device)

        return sag_surface_3d(
            sag_function,
            self.num_iter,
            self.damping,
            self.tol,
            P,
            V,
            tf_in,
            diameter,
            anchors,
            scale,
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
        ScalarTensor,
        Bool[torch.Tensor, ""],
    ]:
        return (
            torch.tensor(10.0, dtype=dtype, device=device),
            torch.tensor(0.5, dtype=dtype, device=device),
            torch.tensor(1.0, dtype=dtype, device=device),
            torch.tensor(
                [[0.1, 0.001, 0.002], [0.1, 0.0, 0.0]], dtype=dtype, device=device
            ),
            torch.tensor(-1.0, dtype=dtype, device=device),
            torch.tensor(False, dtype=torch.bool, device=device),
        )


class XYPolynomial(nn.Module):
    """
    XYPolynomial 3D surface parameterized by:
    - lens diameter
    - signed curvature C
    - conic contant K
    - xypolynomial coefficients

    Represented by a sag function and raytraced by implicit solver
    Support for scale.
    """

    def __init__(
        self,
        diameter: float | ScalarTensor,
        C: float | ScalarTensor | nn.Parameter,
        K: float | ScalarTensor | nn.Parameter,
        coefficients: Sequence[Sequence[float]] | Float[torch.Tensor, "P Q"] | nn.Parameter,
        *,
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
        self.coefficients = init_param(self, "coefficients", coefficients, trainable)
        self.scale = init_param(self, "scale", scale, False)
        self.normalize = init_param(
            self, "normalize", normalize, False, default_dtype=torch.bool
        )
        self.func2d = XYPolynomialSurfaceKernel(num_iter, damping, tol)
        self.func3d = XYPolynomialSurfaceKernel(num_iter, damping, tol)

    def forward(
        self, P: BatchTensor, V: BatchTensor, tf: Tf
    ) -> tuple[BatchTensor, BatchNDTensor, MaskTensor, Tf, Tf]:
        func = self.func2d.apply if P.shape[-1] == 2 else self.func3d.apply
        return func(
            P,
            V,
            tf,
            self.diameter,
            self.C,
            self.K,
            self.coefficients,
            self.scale,
            self.normalize,
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
