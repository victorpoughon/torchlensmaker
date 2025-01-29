import torch
import torch.nn as nn

from torchlensmaker.tensor_manip import cat_optional, cartesian_prod2d, to_tensor
from torchlensmaker.optics import OpticalData

from torchlensmaker.transforms import forward_kinematic
from torchlensmaker.rot2d import rot2d
from torchlensmaker.rot3d import euler_angles_to_matrix

from torchlensmaker.sampling import (
    sampleND,
    Sampler,
    LinearDiskSampler,
    RandomDiskSampler,
)

from typing import Any, Optional

Tensor = torch.Tensor


def unit_vector(dim: int, dtype: torch.dtype) -> Tensor:
    return torch.cat((torch.ones(1, dtype=dtype), torch.zeros(dim - 1, dtype=dtype)))


def rotated_unit_vector(angles: Tensor, dim: int) -> Tensor:
    """
    Rotated unit X vector in 2D or 3D
    angles is batched with shape (N, 2|3)
    """

    dtype = angles.dtype
    N = angles.shape[0]
    if dim == 2:
        unit = torch.tensor([1.0, 0.0], dtype=dtype)
        return rot2d(unit, angles)
    else:
        unit = torch.tensor([1.0, 0.0, 0.0], dtype=dtype)
        thetas = torch.column_stack(
            (
                torch.zeros(N, dtype=dtype),
                angles,
            )
        )
        M = euler_angles_to_matrix(thetas, "XZY").to(
            dtype=dtype
        )  # TODO need to support dtype in euler_angles_to_matrix
        return torch.matmul(M, unit.view(3, 1)).squeeze(-1)


class LightSourceBase(nn.Module):
    def sample_light_source(
        self, sampling: dict[str, Any], dim: int, dtype: torch.dtype
    ) -> tuple[Tensor, Tensor, Tensor, Optional[Tensor]]:
        raise NotImplementedError

    def forward(self, inputs: OpticalData) -> OpticalData:
        dim, dtype = inputs.dim, inputs.dtype

        # Get samples from derived class in local frame
        P, V, rays_base, rays_object = self.sample_light_source(
            inputs.sampling, dim, dtype
        )

        # Apply kinematic transform
        tf = forward_kinematic(inputs.transforms)
        P = tf.direct_points(P)
        V = tf.direct_vectors(V)

        return inputs.replace(
            P=torch.cat((inputs.P, P), dim=0),
            V=torch.cat((inputs.V, V), dim=0),
            rays_base=cat_optional(inputs.rays_base, rays_base),
            rays_object=cat_optional(inputs.rays_object, rays_object),
        )


class PointSourceAtInfinity(LightSourceBase):
    def __init__(self, beam_diameter: float):
        """
        Args:
            beam_diameter: diameter of the beam of light
            angle_offset: incidence angle of the beam (in degrees)
        """
        super().__init__()
        self.beam_diameter: Tensor = to_tensor(beam_diameter)

    def sample_light_source(
        self, sampling: dict[str, Any], dim: int, dtype: torch.dtype
    ) -> tuple[Tensor, Tensor, Tensor, Optional[Tensor]]:

        # Sample coordinates other than X on a disk
        NX = sampleND(
            sampling.get("sampler", None),
            sampling["base"],
            self.beam_diameter,
            dim - 1,
            dtype,
        )

        # Make the rays P + tV
        P = torch.column_stack((torch.zeros(NX.shape[0], dtype=dtype), NX))
        V = torch.tile(unit_vector(dim, dtype), (P.shape[0], 1))

        return P, V, NX, None


class PointSource(LightSourceBase):
    def __init__(self, beam_angular_size: float):
        super().__init__()

        self.beam_angular_size = torch.deg2rad(to_tensor(beam_angular_size))

    def sample_light_source(
        self, sampling: dict[str, Any], dim: int, dtype: torch.dtype
    ) -> tuple[Tensor, Tensor, Tensor, Optional[Tensor]]:

        # Sample angular direction
        angles = sampleND(
            sampling.get("sampler", None),
            sampling["base"],
            self.beam_angular_size,
            dim - 1,
            dtype,
        )

        V = rotated_unit_vector(angles, dim)
        P = torch.zeros_like(V)

        return P, V, angles, None


class ObjectAtInfinity(LightSourceBase):
    def __init__(self, beam_diameter: float, angular_size: float):
        super().__init__()
        self.beam_diameter: Tensor = to_tensor(beam_diameter)
        self.angular_size: Tensor = torch.deg2rad(to_tensor(angular_size))

    def sample_light_source(
        self, sampling: dict[str, Any], dim: int, dtype: torch.dtype
    ) -> tuple[Tensor, Tensor, Tensor, Optional[Tensor]]:

        # Sample coordinates other than X on a disk
        NX = sampleND(
            sampling.get("sampler", None),
            sampling["base"],
            self.beam_diameter,
            dim - 1,
            dtype,
        )
        P = torch.column_stack((torch.zeros(NX.shape[0], dtype=dtype), NX))

        # Sample angular direction
        angles = sampleND(
            sampling.get("sampler", None),
            sampling["object"],
            self.angular_size,
            dim - 1,
            dtype,
        )
        V = rotated_unit_vector(angles, dim)

        # Cartesian product
        P, V = cartesian_prod2d(P, V)
        NX, angles = cartesian_prod2d(NX, angles)

        return P, V, NX, angles


class Object(LightSourceBase):
    def __init__(self, beam_angular_size: float, object_diameter: float):
        super().__init__()
        self.beam_angular_size: Tensor = torch.deg2rad(to_tensor(beam_angular_size))
        self.object_diameter: Tensor = to_tensor(object_diameter)

    def sample_light_source(
        self, sampling: dict[str, Any], dim: int, dtype: torch.dtype
    ) -> tuple[Tensor, Tensor, Tensor, Optional[Tensor]]:

        # Sample coordinates other than X on a disk
        NX = sampleND(
            sampling.get("sampler", None),
            sampling["object"],
            self.object_diameter,
            dim - 1,
            dtype,
        )
        P = torch.column_stack((torch.zeros(NX.shape[0], dtype=dtype), NX))

        # Sample angular direction
        angles = sampleND(
            sampling.get("sampler", None),
            sampling["base"],
            self.beam_angular_size,
            dim - 1,
            dtype,
        )
        V = rotated_unit_vector(angles, dim)

        # Cartesian product
        P, V = cartesian_prod2d(P, V)
        NX, angles = cartesian_prod2d(NX, angles)

        return P, V, angles, NX
