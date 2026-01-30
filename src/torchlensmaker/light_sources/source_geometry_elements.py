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

from jaxtyping import Float

from torchlensmaker.core.tensor_manip import to_tensor

from .source_geometry_kernels import (
    ObjectGeometry2DKernel,
    ObjectGeometry3DKernel,
)


class ObjectGeometry2D(nn.Module):
    def __init__(
        self,
        beam_angular_size: Float[torch.Tensor, ""] | float,
        object_diameter: Float[torch.Tensor, ""] | float,
        wavelength: tuple[int | float, int | float] | int | float = 500,
    ):
        super().__init__()
        self.beam_angular_size = to_tensor(beam_angular_size)
        self.object_diameter = to_tensor(object_diameter)

        if isinstance(wavelength, (int, float)):
            self.wavelength_lower = to_tensor(wavelength)
            self.wavelength_upper = to_tensor(wavelength)
        elif isinstance(wavelength, (tuple, list)):
            self.wavelength_lower = to_tensor(wavelength[0])
            self.wavelength_upper = to_tensor(wavelength[1])
        else:
            raise RuntimeError(
                f"wavelength arg should be a number or a pair of numbers, got {wavelength}"
            )

        self.kernel = ObjectGeometry2DKernel()

    def domain(self) -> dict[str, list[float]]:
        A = self.beam_angular_size.item() / 2
        B = self.object_diameter.item() / 2
        return {
            "base": [-A, A],
            "object": [-B, B],
            "wavelength": [self.wavelength_lower.item(), self.wavelength_upper.item()],
        }

    def forward(
        self,
        pupil_samples: Float[torch.Tensor, " Np"],
        field_samples: Float[torch.Tensor, " Nf"],
        wavelength_samples: Float[torch.Tensor, " Nw"],
    ) -> tuple[
        Float[torch.Tensor, "N 2"],
        Float[torch.Tensor, "N 2"],
        Float[torch.Tensor, " N"],
        Float[torch.Tensor, " N"],
        Float[torch.Tensor, " N"],
    ]:
        P, V, W, psamples, fsamples = self.kernel.forward(
            angular_samples=pupil_samples,
            spatial_samples=field_samples,
            wavelength_samples=wavelength_samples,
            angular_diameter=torch.deg2rad(self.beam_angular_size),
            spatial_diameter=self.object_diameter,
            wavelength_lower=self.wavelength_lower,
            wavelength_upper=self.wavelength_upper,
        )

        return P, V, W, torch.rad2deg(psamples), fsamples


class ObjectAtInfinityGeometry2D(nn.Module):
    def __init__(
        self,
        beam_diameter: Float[torch.Tensor, ""] | float,
        angular_size: Float[torch.Tensor, ""] | float,
        wavelength: tuple[int | float, int | float] | int | float = 500,
    ):
        super().__init__()
        self.beam_diameter = to_tensor(beam_diameter)
        self.angular_size = to_tensor(angular_size)

        if isinstance(wavelength, (int, float)):
            self.wavelength_lower = to_tensor(wavelength)
            self.wavelength_upper = to_tensor(wavelength)
        elif isinstance(wavelength, (tuple, list)):
            self.wavelength_lower = to_tensor(wavelength[0])
            self.wavelength_upper = to_tensor(wavelength[1])
        else:
            raise RuntimeError(
                f"wavelength arg should be a number or a pair of numbers, got {wavelength}"
            )

        self.kernel = ObjectGeometry2DKernel()

    def domain(self) -> dict[str, list[float]]:
        A = self.beam_diameter.item() / 2
        B = self.angular_size.item() / 2
        return {
            "base": [-A, A],
            "object": [-B, B],
            "wavelength": [self.wavelength_lower.item(), self.wavelength_upper.item()],
        }

    def forward(
        self,
        pupil_samples: Float[torch.Tensor, " Np"],
        field_samples: Float[torch.Tensor, " Nf"],
        wavelength_samples: Float[torch.Tensor, " Nw"],
    ) -> tuple[
        Float[torch.Tensor, "N 2"],
        Float[torch.Tensor, "N 2"],
        Float[torch.Tensor, " N"],
        Float[torch.Tensor, " N"],
        Float[torch.Tensor, " N"],
    ]:
        P, V, W, angular_samples, spatial_samples = self.kernel.forward(
            angular_samples=field_samples,
            spatial_samples=pupil_samples,
            wavelength_samples=wavelength_samples,
            angular_diameter=torch.deg2rad(self.angular_size),
            spatial_diameter=self.beam_diameter,
            wavelength_lower=self.wavelength_lower,
            wavelength_upper=self.wavelength_upper,
        )

        return P, V, W, spatial_samples, torch.rad2deg(angular_samples)


class ObjectGeometry3D(nn.Module):
    def __init__(
        self,
        beam_angular_size: Float[torch.Tensor, ""] | float,
        object_diameter: Float[torch.Tensor, ""] | float,
        wavelength: tuple[int | float, int | float] | int | float = 500,
    ):
        super().__init__()
        self.beam_angular_size = to_tensor(beam_angular_size)
        self.object_diameter = to_tensor(object_diameter)

        if isinstance(wavelength, (int, float)):
            self.wavelength_lower = to_tensor(wavelength)
            self.wavelength_upper = to_tensor(wavelength)
        elif isinstance(wavelength, (tuple, list)):
            self.wavelength_lower = to_tensor(wavelength[0])
            self.wavelength_upper = to_tensor(wavelength[1])
        else:
            raise RuntimeError(
                f"wavelength arg should be a number or a pair of numbers, got {wavelength}"
            )

        self.kernel = ObjectGeometry3DKernel()

    def domain(self) -> dict[str, list[float]]:
        return {
            "wavelength": [self.wavelength_lower.item(), self.wavelength_upper.item()],
        }

    def forward(
        self,
        pupil_samples: Float[torch.Tensor, " Np 2"],
        field_samples: Float[torch.Tensor, " Nf 2"],
        wavelength_samples: Float[torch.Tensor, " Nw"],
    ) -> tuple[
        Float[torch.Tensor, "N 3"],
        Float[torch.Tensor, "N 3"],
        Float[torch.Tensor, " N"],
        Float[torch.Tensor, "N 2"],
        Float[torch.Tensor, "N 2"],
    ]:
        P, V, W, psamples, fsamples = self.kernel.forward(
            angular_samples=pupil_samples,
            spatial_samples=field_samples,
            wavelength_samples=wavelength_samples,
            angular_diameter=torch.deg2rad(self.beam_angular_size),
            spatial_diameter=self.object_diameter,
            wavelength_lower=self.wavelength_lower,
            wavelength_upper=self.wavelength_upper,
        )

        return P, V, W, torch.rad2deg(psamples), fsamples


class ObjectAtInfinityGeometry3D(nn.Module):
    def __init__(
        self,
        beam_diameter: Float[torch.Tensor, ""] | float,
        angular_size: Float[torch.Tensor, ""] | float,
        wavelength: tuple[int | float, int | float] | int | float = 500,
    ):
        super().__init__()
        self.beam_diameter = to_tensor(beam_diameter)
        self.angular_size = to_tensor(angular_size)

        if isinstance(wavelength, (int, float)):
            self.wavelength_lower = to_tensor(wavelength)
            self.wavelength_upper = to_tensor(wavelength)
        elif isinstance(wavelength, (tuple, list)):
            self.wavelength_lower = to_tensor(wavelength[0])
            self.wavelength_upper = to_tensor(wavelength[1])
        else:
            raise RuntimeError(
                f"wavelength arg should be a number or a pair of numbers, got {wavelength}"
            )

        self.kernel = ObjectGeometry3DKernel()

    def domain(self) -> dict[str, list[float]]:
        return {
            "wavelength": [self.wavelength_lower.item(), self.wavelength_upper.item()],
        }

    def forward(
        self,
        pupil_samples: Float[torch.Tensor, "Np 2"],
        field_samples: Float[torch.Tensor, "Nf 2"],
        wavelength_samples: Float[torch.Tensor, " Nw"],
    ) -> tuple[
        Float[torch.Tensor, "N 3"],
        Float[torch.Tensor, "N 3"],
        Float[torch.Tensor, " N"],
        Float[torch.Tensor, "N 2"],
        Float[torch.Tensor, "N 2"],
    ]:
        P, V, W, angular_samples, spatial_samples = self.kernel.forward(
            angular_samples=field_samples,
            spatial_samples=pupil_samples,
            wavelength_samples=wavelength_samples,
            angular_diameter=torch.deg2rad(self.angular_size),
            spatial_diameter=self.beam_diameter,
            wavelength_lower=self.wavelength_lower,
            wavelength_upper=self.wavelength_upper,
        )

        return P, V, W, spatial_samples, torch.rad2deg(angular_samples)
