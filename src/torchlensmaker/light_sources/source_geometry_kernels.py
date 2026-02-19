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

from typing import TypeAlias, Sequence
from jaxtyping import Float, Int
import torch
from torchlensmaker.core.functional_kernel import FunctionalKernel

from torchlensmaker.core.tensor_manip import (
    meshgrid_flat,
    meshgrid2d_flat3,
)

from torchlensmaker.core.geometry import rotate_x_zy

from torchlensmaker.types import BatchTensor, Batch2DTensor, Batch3DTensor, ScalarTensor


class ObjectGeometry2DKernel(FunctionalKernel):
    """
    An object in 2D represented by:
        - an angular diameter
        - a spatial diameter
        - a wavelength bandwith

    When using this kernel, the mapping between angular/spatial and pupil/field
    will determine if the object is in the near field or at infinity.
    """

    inputs = {
        "angular_samples": BatchTensor,  # (Na,) normalized [-1, 1] samples in the angular dimension
        "spatial_samples": BatchTensor,  # (Ns,) normalized [-1, 1] samples in the spatial dimension
        "wavelength_samples": BatchTensor,  # (Nw,) normalized [-1, 1] samples in the wavelength dimension
    }

    params = {
        "angular_diameter": ScalarTensor,  # angular diameter in radians
        "spatial_diameter": ScalarTensor,  # spatial diameter in length units
        "wavelength_lower": ScalarTensor,  # lower bound for the wavelength domain
        "wavelength_upper": ScalarTensor,  # upper bound for the wavelength domain
    }

    outputs = {
        "P": Batch2DTensor,  # (N, 2) rays origins
        "V": Batch2DTensor,  # (N, 2) rays direction
        "W": BatchTensor,  # (N,) rays wavelength
        "angular_coordinates": BatchTensor,  # (N,) rays angular coordinates
        "spatial_coordinates": BatchTensor,  # (N,) rays spatial coordinates
    }

    def apply(
        self,
        angular_samples: Float[torch.Tensor, " Na"],
        spatial_samples: Float[torch.Tensor, " Ns"],
        wavelength_samples: Float[torch.Tensor, " Nw"],
        angular_diameter: Float[torch.Tensor, ""],
        spatial_diameter: Float[torch.Tensor, ""],
        wavelength_lower: Float[torch.Tensor, ""],
        wavelength_upper: Float[torch.Tensor, ""],
    ) -> tuple[
        Float[torch.Tensor, "N 2"],
        Float[torch.Tensor, "N 2"],
        Float[torch.Tensor, " N"],
        Float[torch.Tensor, " N"],
        Float[torch.Tensor, " N"],
    ]:
        # pupil coordinates are the pupil samples over the beam angular size
        angular_coords = angular_samples * angular_diameter / 2

        # field coordinates are the field samples over the object diameter
        spatial_coords = spatial_samples * spatial_diameter / 2

        # wavelength coordinates are scaled over the wavelength domain
        bandwith = wavelength_upper - wavelength_lower
        wavel_coords = wavelength_lower + bandwith * (wavelength_samples + 1) / 2

        # Generate all possible combinations of ray coordinates with a triple meshgrid
        spatial_coords_full, angular_coords_full, wavel_coords_full = meshgrid_flat(
            spatial_coords, angular_coords, wavel_coords
        )

        # Convert pupil and field to physical space
        P = torch.stack(
            (torch.zeros_like(spatial_coords_full), spatial_coords_full), dim=-1
        )
        V = torch.stack(
            (torch.cos(angular_coords_full), torch.sin(angular_coords_full)), dim=-1
        )

        return (
            P,
            V,
            wavel_coords_full,
            angular_coords_full,
            spatial_coords_full,
        )

    def example_inputs(
        self, dtype: torch.dtype, device: torch.device
    ) -> tuple[torch.Tensor, ...]:
        pupil_samples = torch.linspace(-1, 1, 10, dtype=dtype, device=device)
        field_samples = torch.linspace(-1, 1, 10, dtype=dtype, device=device)
        wavelength_samples = torch.linspace(-1, 1, 10, dtype=dtype, device=device)

        return (pupil_samples, field_samples, wavelength_samples)

    def example_params(
        self, dtype: torch.dtype, device: torch.device
    ) -> tuple[torch.Tensor, ...]:
        return (
            torch.tensor(15, dtype=dtype),
            torch.tensor(5, dtype=dtype),
            torch.tensor(400, dtype=dtype),
            torch.tensor(800, dtype=dtype),
        )


class ObjectGeometry3DKernel(FunctionalKernel):
    """
    An object in 3D represented by a disk in spatial and angular dimensions, and
    a band in the spectral dimension.

    When using this kernel, the mapping between angular/spatial and pupil/field
    will determine if the object is in the near field or at infinity.
    """

    inputs = {
        "angular_samples": Batch2DTensor,  # (Np, 2) normalized [-1, 1] samples in the angular dimension
        "spatial_samples": Batch2DTensor,  # (Nf, 2) normalized [-1, 1] samples in the spatial dimension
        "wavelength_samples": BatchTensor,  # (Nw,) normalized [-1, 1] samples in the wavelength dimension
    }

    params = {
        "angular_diameter": ScalarTensor,  # angular diameter in radians
        "spatial_diameter": ScalarTensor,  # spatial diameter in length units
        "wavelength_lower": ScalarTensor,  # lower bound for the wavelength domain
        "wavelength_upper": ScalarTensor,  # upper bound for the wavelength domain
    }

    outputs = {
        "P": Batch3DTensor,  # (N, 3) rays origins
        "V": Batch3DTensor,  # (N, 3) rays direction
        "W": BatchTensor,  # (N,) rays wavelength
        "angular_coordinates": Batch2DTensor,  # (N, 2) rays angular coordinates
        "spatial_coordinates": Batch2DTensor,  # (N, 2) rays spatial coordinates
    }

    def apply(
        self,
        angular_samples: Float[torch.Tensor, "Np 2"],
        spatial_samples: Float[torch.Tensor, "Nf 2"],
        wavelength_samples: Float[torch.Tensor, " Nw"],
        angular_diameter: Float[torch.Tensor, ""],
        spatial_diameter: Float[torch.Tensor, ""],
        wavelength_lower: Float[torch.Tensor, ""],
        wavelength_upper: Float[torch.Tensor, ""],
    ) -> tuple[
        Float[torch.Tensor, "N 3"],
        Float[torch.Tensor, "N 3"],
        Float[torch.Tensor, " N"],
        Float[torch.Tensor, "N 2"],
        Float[torch.Tensor, "N 2"],
    ]:
        # pupil coordinates are the pupil samples over the beam angular size
        angular_coords = angular_samples * angular_diameter / 2

        # field coordinates are the field samples over the object diameter
        spatial_coords = spatial_samples * spatial_diameter / 2

        # wavelength coordinates are scaled over the wavelength domain
        bandwith = wavelength_upper - wavelength_lower
        wavel_coords = wavelength_lower + bandwith * (wavelength_samples + 1) / 2

        # Generate all possible combinations of ray coordinates with a triple meshgrid
        spatial_coords_full, angular_coords_full, wavel_coords_full = meshgrid2d_flat3(
            spatial_coords, angular_coords, wavel_coords
        )

        # Convert field to physical space by adding X=0 to the YZ plane samples
        Px = torch.zeros(
            (spatial_coords_full.shape[0], 1),
            dtype=spatial_coords_full.dtype,
            device=spatial_coords_full.device,
        )
        P = torch.cat((Px, spatial_coords_full), dim=-1)

        # Convert pupil to physical space by rotating the unit X vector
        V = rotate_x_zy(angular_coords_full)

        return (
            P,
            V,
            wavel_coords_full,
            angular_coords_full,
            spatial_coords_full,
        )

    def example_inputs(
        self, dtype: torch.dtype, device: torch.device
    ) -> tuple[torch.Tensor, ...]:
        pupil_samples = torch.tensor(
            [
                [0, 0],
                [-1, 0],
                [0, -1],
                [-1, -1],
                [0, 1],
                [1, 0],
                [1, 1],
                [-1, 1],
                [1, -1],
            ],
            dtype=dtype,
            device=device,
        )
        field_samples = torch.tensor(
            [
                [0, 0],
                [-1, 0],
                [0, -1],
                [-1, -1],
                [0, 1],
                [1, 0],
                [1, 1],
                [-1, 1],
                [1, -1],
            ],
            dtype=dtype,
            device=device,
        )
        wavelength_samples = torch.linspace(-1, 1, 10, dtype=dtype, device=device)

        return (pupil_samples, field_samples, wavelength_samples)

    def example_params(
        self, dtype: torch.dtype, device: torch.device
    ) -> tuple[torch.Tensor, ...]:
        return (
            torch.tensor(15, dtype=dtype),
            torch.tensor(5, dtype=dtype),
            torch.tensor(400, dtype=dtype),
            torch.tensor(800, dtype=dtype),
        )
