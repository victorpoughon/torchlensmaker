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

from torchlensmaker.core.tensor_manip import cartesian_prod2d, meshgrid_flat


class ObjectGeometry2DKernel(FunctionalKernel):
    """
    Object in the near field
    """

    input_names = [
        "pupil_samples",  # (Np,) normalized [-1, 1] samples in the pupil dimension
        "field_samples",  # (Nf,) normalized [-1, 1] samples in the field dimension
        "wavelength_samples",  # (Nw,) normalized [-1, 1] samples in the wavelength dimension
    ]

    param_names = [
        "beam_angular_size",  # angular size of the pupil beam in degrees
        "object_diameter",  # diameter of the object in length units
        "wavelength_lower",  # lower bound for the wavelength domain
        "wavelength_upper",  # upper bound for the wavelength domain
    ]

    output_names = [
        "P",  # (N, 2) rays origins
        "V",  # (N, 2) rays direction
        "W",  # (N,) rays wavelength
        "pupil_coordinates",  # (N,) rays pupil coordinates
        "field_coordinates",  # (N,) rays field coordinates
    ]

    @staticmethod
    def forward(
        pupil_samples: Float[torch.Tensor, " Np"],
        field_samples: Float[torch.Tensor, " Nf"],
        wavelength_samples: Float[torch.Tensor, " Nw"],
        beam_angular_size: Float[torch.Tensor, ""],
        object_diameter: Float[torch.Tensor, ""],
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
        pupil_coords = pupil_samples * torch.deg2rad(beam_angular_size / 2)

        # field coordinates are the field samples over the object diameter
        field_coords = field_samples * object_diameter / 2

        # wavelength coordinates are scaled over the wavelength domain
        bandwith = wavelength_upper - wavelength_lower
        wavel_coords = wavelength_lower + bandwith * (wavelength_samples + 1) / 2

        # Generate all possible combinations of ray coordinates with a triple meshgrid
        field_coords_full, pupil_coords_full, wavel_coords_full = meshgrid_flat(
            field_coords, pupil_coords, wavel_coords
        )

        # Convert pupil and field to physical space
        P = torch.stack(
            (torch.zeros_like(field_coords_full), field_coords_full), dim=-1
        )
        V = torch.stack(
            (torch.cos(pupil_coords_full), torch.sin(pupil_coords_full)), dim=-1
        )

        return P, V, wavel_coords_full, torch.rad2deg(pupil_coords_full), field_coords_full

    @staticmethod
    def example_inputs(
        dtype: torch.dtype, device: torch.device
    ) -> tuple[torch.Tensor, ...]:
        pupil_samples = torch.linspace(-1, 1, 10, dtype=dtype, device=device)
        field_samples = torch.linspace(-1, 1, 10, dtype=dtype, device=device)
        wavelength_samples = torch.linspace(-1, 1, 10, dtype=dtype, device=device)

        return (pupil_samples, field_samples, wavelength_samples)

    @staticmethod
    def example_params(
        dtype: torch.dtype, device: torch.device
    ) -> tuple[torch.Tensor, ...]:
        return (
            torch.tensor(15, dtype=dtype),
            torch.tensor(5, dtype=dtype),
            torch.tensor(400, dtype=dtype),
            torch.tensor(800, dtype=dtype),
        )


class ObjectAtInfinityGeometry2DKernel(FunctionalKernel):
    """
    A 2D object at infinity
    """

    input_names = [
        "pupil_samples",  # (Np,) normalized [-1, 1] samples in the pupil dimension
        "field_samples",  # (Nf,) normalized [-1, 1] samples in the field dimension
        "wavelength_samples",  # (Nw,) normalized [-1, 1] samples in the wavelength dimension
    ]

    param_names = [
        "beam_diameter",  # diameter of the beam in length units
        "angular_size",  # object apparent angular size in degrees
        "wavelength_lower",  # lower bound for the wavelength domain
        "wavelength_upper",  # upper bound for the wavelength domain
    ]

    output_names = [
        "P",  # (N, 2) rays origins
        "V",  # (N, 2) rays direction
        "W",  # (N,) rays wavelength
        "pupil_coordinates",  # (N,) rays pupil coordinates
        "field_coordinates",  # (N,) rays field coordinates
    ]

    @staticmethod
    def forward(
        pupil_samples: Float[torch.Tensor, " Np"],
        field_samples: Float[torch.Tensor, " Nf"],
        wavelength_samples: Float[torch.Tensor, " Nw"],
        beam_diameter: Float[torch.Tensor, ""],
        angular_size: Float[torch.Tensor, ""],
        wavelength_lower: Float[torch.Tensor, ""],
        wavelength_upper: Float[torch.Tensor, ""],
    ) -> tuple[
        Float[torch.Tensor, "N 2"],
        Float[torch.Tensor, "N 2"],
        Float[torch.Tensor, " N"],
        Float[torch.Tensor, " N"],
        Float[torch.Tensor, " N"],
    ]:
        # pupil coordinates are the pupil samples over the beam diameter
        pupil_coords = pupil_samples * beam_diameter / 2

        # field coordinates are the field samples over the object angular size
        field_coords = field_samples * torch.deg2rad(angular_size / 2)

        # wavelength coordinates are scaled over the wavelength domain
        bandwith = wavelength_upper - wavelength_lower
        wavel_coords = wavelength_lower + bandwith * (wavelength_samples + 1) / 2

        # Generate all possible combinations of ray coordinates with a triple meshgrid
        field_coords_full, pupil_coords_full, wavel_coords_full = meshgrid_flat(
            field_coords, pupil_coords, wavel_coords
        )

        # Convert pupil and field to physical space
        P = torch.stack(
            (torch.zeros_like(pupil_coords_full), pupil_coords_full), dim=-1
        )
        V = torch.stack(
            (torch.cos(field_coords_full), torch.sin(field_coords_full)), dim=-1
        )

        return P, V, wavel_coords_full, pupil_coords_full, torch.rad2deg(field_coords_full)

    @staticmethod
    def example_inputs(
        dtype: torch.dtype, device: torch.device
    ) -> tuple[torch.Tensor, ...]:
        pupil_samples = torch.linspace(-1, 1, 10, dtype=dtype, device=device)
        field_samples = torch.linspace(-1, 1, 10, dtype=dtype, device=device)
        wavelength_samples = torch.linspace(-1, 1, 10, dtype=dtype, device=device)

        return (pupil_samples, field_samples, wavelength_samples)

    @staticmethod
    def example_params(
        dtype: torch.dtype, device: torch.device
    ) -> tuple[torch.Tensor, ...]:
        return (
            torch.tensor(15, dtype=dtype),
            torch.tensor(5, dtype=dtype),
            torch.tensor(400, dtype=dtype),
            torch.tensor(800, dtype=dtype),
        )
