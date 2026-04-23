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

from typing import Sequence, TypeAlias

import torch
from jaxtyping import Float, Int

from torchlensmaker.core.functional_kernel import FunctionalKernel
from torchlensmaker.core.geometry import rotate_x_zy
from torchlensmaker.core.tensor_manip import (
    meshgrid2d_flat3,
    meshgrid_flat,
)
from torchlensmaker.kinematics.homogeneous_geometry import (
    hom_identity_2d,
    hom_identity_3d,
    transform_rays,
)
from torchlensmaker.types import (
    Batch2DTensor,
    Batch3DTensor,
    BatchTensor,
    HomMatrix,
    IndexTensor,
    ScalarTensor,
)


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
        "tf": HomMatrix,  # kinematic direct transform applied to the light source
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
        "angular_coordinates": Batch2DTensor,  # (N,) rays angular coordinates
        "spatial_coordinates": Batch2DTensor,  # (N,) rays spatial coordinates
        "wavelength_coordinates": BatchTensor,  # (N,) rays wavelength
        "angular_idx": IndexTensor,  # (N,) index of angular samples
        "spatial_idx": IndexTensor,  # (N,) index of spatial samples
        "wavelength_idx": IndexTensor,  # (N,) index of wavelength samples
    }

    dynamic_shapes = {
        "angular_samples": {0: "N_angular"},
        "spatial_samples": {0: "N_spatial"},
        "wavelength_samples": {0: "N_wavelength"},
    }

    # seems to be some issue with meshgrid in some cases
    # maybe swithing to arange would help, to be investigated
    export_legacy = True

    def apply(
        self,
        tf: HomMatrix,
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
        IndexTensor,
        IndexTensor,
        IndexTensor,
    ]:
        device = tf.device
        # angular coordinates are the angular samples over the angular domain
        angular_coords = angular_samples * angular_diameter / 2
        angular_idx = torch.arange(angular_samples.size(0), device=device)

        # spatial coordinates are the spatial samples over the spatial domain
        spatial_coords = spatial_samples * spatial_diameter / 2
        spatial_idx = torch.arange(spatial_samples.size(0), device=device)

        # wavelength coordinates are scaled over the wavelength domain
        bandwith = wavelength_upper - wavelength_lower
        wavel_coords = wavelength_lower + bandwith * (wavelength_samples + 1) / 2
        wavel_idx = torch.arange(wavelength_samples.size(0), device=device)

        # Generate all possible combinations of ray coordinates with a triple meshgrid
        angular_coords_full, spatial_coords_full, wavel_coords_full = meshgrid_flat(
            angular_coords, spatial_coords, wavel_coords
        )

        # Same for indices
        angular_idx_full, spatial_idx_full, wavel_idx_full = meshgrid_flat(
            angular_idx, spatial_idx, wavel_idx
        )

        # Convert spatial coordinates to physical space by adding X=0 to the YZ plane samples
        P = torch.stack(
            (torch.zeros_like(spatial_coords_full), spatial_coords_full), dim=-1
        )

        # Convert angular coordinates to direction vectors by rotating the unit X vector
        V = torch.stack(
            (torch.cos(angular_coords_full), torch.sin(angular_coords_full)), dim=-1
        )

        # Apply kinematic transform
        P, V = transform_rays(tf, P, V)

        return (
            P,
            V,
            angular_coords_full,
            spatial_coords_full,
            wavel_coords_full,
            angular_idx_full,
            spatial_idx_full,
            wavel_idx_full,
        )

    def example_inputs(
        self, dtype: torch.dtype, device: torch.device
    ) -> tuple[torch.Tensor, ...]:
        tf = hom_identity_2d(dtype, device).direct
        pupil_samples = torch.linspace(-1, 1, 10, dtype=dtype, device=device)
        field_samples = torch.linspace(-1, 1, 10, dtype=dtype, device=device)
        wavelength_samples = torch.linspace(-1, 1, 10, dtype=dtype, device=device)

        return (tf, pupil_samples, field_samples, wavelength_samples)

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
        "tf": HomMatrix,  # kinematic direct transform applied to the light source
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
        "angular_coordinates": Batch2DTensor,  # (N, 2) rays angular coordinates
        "spatial_coordinates": Batch2DTensor,  # (N, 2) rays spatial coordinates
        "wavelength_coordinates": BatchTensor,  # (N,) rays wavelength
        "angular_idx": IndexTensor,  # (N,) index of angular samples
        "spatial_idx": IndexTensor,  # (N,) index of spatial samples
        "wavelength_idx": IndexTensor,  # (N,) index of wavelength samples
    }

    dynamic_shapes = {
        "angular_samples": {0: "N_angular"},
        "spatial_samples": {0: "N_spatial"},
        "wavelength_samples": {0: "N_wavelength"},
    }

    # seems to be some issue with meshgrid in some cases
    # maybe swithing to arange would help, to be investigated
    export_legacy = True

    def apply(
        self,
        tf: HomMatrix,
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
        IndexTensor,
        IndexTensor,
        IndexTensor,
    ]:
        device = tf.device
        # angular coordinates are the angular samples over the angular domain
        angular_coords = angular_samples * angular_diameter / 2
        angular_idx = torch.arange(angular_samples.size(0), device=device)

        # spatial coordinates are the spatial samples over the spatial domain
        spatial_coords = spatial_samples * spatial_diameter / 2
        spatial_idx = torch.arange(spatial_samples.size(0), device=device)

        # wavelength coordinates are scaled over the wavelength domain
        bandwith = wavelength_upper - wavelength_lower
        wavel_coords = wavelength_lower + bandwith * (wavelength_samples + 1) / 2
        wavel_idx = torch.arange(wavelength_samples.size(0), device=device)

        # Generate all possible combinations of ray coordinates with a triple meshgrid
        angular_coords_full, spatial_coords_full, wavel_coords_full = meshgrid2d_flat3(
            angular_coords, spatial_coords, wavel_coords
        )

        # Same for indices
        angular_idx_full, spatial_idx_full, wavel_idx_full = meshgrid2d_flat3(
            angular_idx, spatial_idx, wavel_idx
        )

        # Convert spatial coordinates to physical space by adding X=0 to the YZ plane samples
        Px = torch.zeros(
            (spatial_coords_full.shape[0], 1),
            dtype=spatial_coords_full.dtype,
            device=spatial_coords_full.device,
        )
        P = torch.cat((Px, spatial_coords_full), dim=-1)

        # Convert angular coordinates to direction vectors by rotating the unit X vector
        V = rotate_x_zy(angular_coords_full)

        # Apply kinematic transform
        P, V = transform_rays(tf, P, V)

        return (
            P,
            V,
            angular_coords_full,
            spatial_coords_full,
            wavel_coords_full,
            angular_idx_full,
            spatial_idx_full,
            wavel_idx_full,
        )

    def example_inputs(
        self, dtype: torch.dtype, device: torch.device
    ) -> tuple[torch.Tensor, ...]:
        tf = hom_identity_3d(dtype, device).direct
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

        return (tf, pupil_samples, field_samples, wavelength_samples)

    def example_params(
        self, dtype: torch.dtype, device: torch.device
    ) -> tuple[torch.Tensor, ...]:
        return (
            torch.tensor(15, dtype=dtype),
            torch.tensor(5, dtype=dtype),
            torch.tensor(400, dtype=dtype),
            torch.tensor(800, dtype=dtype),
        )
