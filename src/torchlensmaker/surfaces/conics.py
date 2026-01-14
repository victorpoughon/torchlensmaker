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

import torch
import torch.nn as nn

from torchlensmaker.core.tensor_manip import to_tensor
from torchlensmaker.surfaces.sag_surface import SagSurface
from torchlensmaker.core.sag_functions import (
    Spherical,
    Parabolic,
    Aspheric,
    Conical,
    SagSum,
)
from torchlensmaker.core.collision_detection import (
    CollisionMethod,
    default_collision_method,
)


from typing import TypeAlias

Tensor: TypeAlias = torch.Tensor


class Sphere(SagSurface):
    """
    A section of a sphere, parameterized by signed curvature.
    Curvature is the inverse of radius: C = 1/R.

    This parameterization is useful because it enables clean representation of
    an infinite radius section of sphere (which is really a plane), and also
    enables changing the sign of C during optimization.

    In 2D, this surface is an arc of circle.
    In 3D, this surface is a section of a sphere (wikipedia calls it a "spherical cap")

    For high curvature arcs (close to a half circle), it's better to use the
    SphereR class which uses radius parameterization and polar distance
    functions. In fact this class cannot represent an exact half circle (R =
    D/2) due to the gradient becoming infinite, use SphereR instead.
    """

    def __init__(
        self,
        diameter: float,
        R: int | float | nn.Parameter | None = None,
        C: int | float | nn.Parameter | None = None,
        normalize: bool = False,
        collision_method: CollisionMethod = default_collision_method,
        dtype: torch.dtype | None = None,
    ):
        if dtype is None:
            dtype = torch.get_default_dtype()
        
        if (R is not None and C is not None) or (R is None and C is None):
            raise RuntimeError(
                "Sphere must be initialized with exactly one of R (radius) or C (curvature)."
            )

        C_tensor: torch.Tensor | nn.Parameter
        if C is None and R is not None:
            if isinstance(R, nn.Parameter):
                C_tensor = nn.Parameter(torch.tensor(1.0 / R.item(), dtype=R.dtype))
            else:
                C_tensor = torch.as_tensor(1.0 / R, dtype=dtype)
        else:
            if isinstance(C, nn.Parameter):
                C_tensor = C
            else:
                C_tensor = torch.as_tensor(C, dtype=dtype)

        # Domain error check
        tau = diameter / 2
        C_unnormed = C_tensor / tau if normalize else C_tensor
        if torch.abs(1.0 / C_unnormed) <= tau:
            raise RuntimeError(
                f"Sphere radius must be strictly greater than half the surface diameter (D/2={diameter / 2}) "
                f"(To model an exact half-sphere, use SphereR)."
            )

        assert C_tensor.dim() == 0
        assert C_tensor.dtype == dtype

        self._sag = Spherical(C_tensor, normalize)

        super().__init__(diameter, self._sag, collision_method, dtype)

    def radius(self) -> float:
        "Utility function to get radius from internal curvature"
        return torch.div(
            torch.tensor(1.0, dtype=self.dtype), self._sag.C
        ).item()


class Parabola(SagSurface):
    "Sag surface for a parabola $X = A R^2$"

    def __init__(
        self,
        diameter: float,
        A: int | float | nn.Parameter,
        normalize: bool = False,
        dtype: torch.dtype = torch.float64,
    ):
        if isinstance(A, torch.Tensor):
            assert A.dtype == dtype
        A_tensor = to_tensor(A, default_dtype=dtype)
        self._sag = Parabolic(A_tensor, normalize)
        super().__init__(diameter, self._sag, dtype=dtype)

    @property
    def A(self) -> torch.Tensor:
        return self._sag.A


# TODO
class Conic(SagSurface): ...


class Asphere(SagSurface):
    """
    Sag surface for the Asphere model

    An asphere is a conical base + aspheric coefficients
    """

    def __init__(
        self,
        diameter: float,
        R: int | float | nn.Parameter,
        K: int | float | nn.Parameter,
        coefficients: list[int | float] | torch.Tensor,
        normalize_conical: bool = False,
        normalize_aspheric: bool = False,
        collision_method: CollisionMethod = default_collision_method,
        dtype: torch.dtype = torch.float64,
    ):
        if isinstance(R, torch.Tensor):
            assert R.dtype == dtype
            assert R.dim() == 0
        if isinstance(K, torch.Tensor):
            assert K.dtype == dtype
            assert K.dim() == 0
        if isinstance(coefficients, torch.Tensor):
            assert coefficients.dtype == dtype
            assert coefficients.dim() == 1

        C_tensor = to_tensor(1.0 / R, default_dtype=dtype)
        K_tensor = to_tensor(K, default_dtype=dtype)
        coefficients_tensor = to_tensor(coefficients, default_dtype=dtype)

        # This prevents against initializing with out of domain K value,
        # but not getting there during optimization.
        # TODO add an Asphere model reparameterized with softplus
        tau = diameter / 2
        C_unnormed = C_tensor / tau if normalize_conical else C_tensor
        if diameter / 2 >= torch.sqrt(1 / (C_unnormed**2 * (1 + K_tensor))):
            raise ValueError(f"Out of domain asphere parameters {C_tensor} {K_tensor}")

        sag_function = SagSum(
            [
                Conical(C_tensor, K_tensor, normalize_conical),
                Aspheric(coefficients_tensor, normalize_aspheric),
            ]
        )

        super().__init__(diameter, sag_function, collision_method, dtype)
