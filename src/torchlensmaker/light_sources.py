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

from torchlensmaker.materials import MaterialModel, get_material_model


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
    def __init__(self, material: str | MaterialModel = "air"):
        super().__init__()
        self.material = get_material_model(material)

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
            material=self.material,
        )


class RaySource(LightSourceBase):
    def sample_light_source(
        self, sampling: dict[str, Any], dim: int, dtype: torch.dtype
    ) -> tuple[Tensor, Tensor, Tensor, Optional[Tensor]]:

        P = torch.zeros(1, dim, dtype=dtype)
        V = unit_vector(dim, dtype).unsqueeze(0)

        return P, V, None, None


class PointSourceAtInfinity(LightSourceBase):
    def __init__(self, beam_diameter: float, **kwargs: Any):
        """
        Args:
            beam_diameter: diameter of the beam of light
            angle_offset: incidence angle of the beam (in degrees)
        """
        super().__init__(**kwargs)
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
    def __init__(self, beam_angular_size: float, **kwargs: Any):
        super().__init__(**kwargs)

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
    def __init__(self, beam_diameter: float, angular_size: float, **kwargs: Any):
        super().__init__(**kwargs)
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
    def __init__(self, beam_angular_size: float, object_diameter: float, **kwargs: Any):
        super().__init__(**kwargs)
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


class Monochromatic(nn.Module):
    def __init__(self, wavelength: float | int):
        super().__init__()
        self.wavelength = wavelength

    def forward(self, inputs: OpticalData) -> OpticalData:
        if inputs.rays_wavelength is not None:
            raise RuntimeError("Rays already have wavelength data")

        return inputs.replace(
            rays_wavelength=torch.full_like(inputs.P[:, 0], self.wavelength)
        )


def cartesian_wavelength(inputs, ray_var):
    "Add wavelength var by doing cartesian product with existing vars"

    N = inputs.P.shape[0]
    M = ray_var.shape[0]

    new_P = torch.repeat_interleave(inputs.P, M, dim=0)
    new_V = torch.repeat_interleave(inputs.V, M, dim=0)
    new_rays_base = (
        torch.repeat_interleave(inputs.rays_base, M, dim=0)
        if inputs.rays_base is not None
        else None
    )
    new_rays_object = (
        torch.repeat_interleave(inputs.rays_object, M, dim=0)
        if inputs.rays_object is not None
        else None
    )

    new_var = torch.tile(ray_var, (N,))

    return inputs.replace(
        P=new_P,
        V=new_V,
        rays_base=new_rays_base,
        rays_object=new_rays_object,
        rays_wavelength=new_var,
    )


class Multichromatic(nn.Module):
    def __init__(self, wavelengths: list[float | int]):
        super().__init__()
        self.wavelengths = to_tensor(wavelengths)

    def forward(self, inputs: OpticalData) -> OpticalData:
        if inputs.rays_wavelength is not None:
            raise RuntimeError("Rays already have wavelength data")

        chromatic_space = self.wavelengths

        return cartesian_wavelength(inputs, chromatic_space)


class ChromaticRange(nn.Module):
    def __init__(self, wmin: float | int, wmax: float | int):
        super().__init__()
        self.wmin, self.wmax = wmin, wmax

    def forward(self, inputs: OpticalData) -> OpticalData:
        if inputs.rays_wavelength is not None:
            raise RuntimeError(
                "Rays already have wavelength data. Cannot apply ChromaticRange()."
            )

        if "wavelength" not in inputs.sampling:
            raise RuntimeError(
                "Missing 'wavelength' key in sampling configuration. Cannot apply ChromaticRange()."
            )

        # TODO option to offset along the base or object coordinate

        chromatic_space = self.wmin + sampleND(
            inputs.sampling.get("sampler", None),
            inputs.sampling["wavelength"],
            self.wmax - self.wmin,
            1,
            inputs.dtype,
        )

        return cartesian_wavelength(inputs, chromatic_space)
