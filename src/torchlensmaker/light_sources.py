import torch
import torch.nn as nn

from torchlensmaker.core.tensor_manip import (
    cat_optional,
    cartesian_prod2d_optional,
    to_tensor,
)
from torchlensmaker.optics import OpticalData

from torchlensmaker.core.transforms import forward_kinematic
from torchlensmaker.core.rot2d import rot2d
from torchlensmaker.core.rot3d import euler_angles_to_matrix

from torchlensmaker.sampling.samplers import (
    sampleND,
)

from torchlensmaker.materials import MaterialModel, get_material_model


from typing import Any, Optional

Tensor = torch.Tensor


def unit_vector(dim: int, dtype: torch.dtype) -> Tensor:
    "Unit vector along the X axis"
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
        P, V, var_base, var_object = self.sample_light_source(
            inputs.sampling, dim, dtype
        )

        # Cartesian product
        P, V = cartesian_prod2d_optional(P, V)
        rays_base, rays_object = cartesian_prod2d_optional(var_base, var_object)

        # Apply kinematic transform
        tf = forward_kinematic(inputs.transforms)
        P = tf.direct_points(P)
        V = tf.direct_vectors(V)

        return inputs.replace(
            P=torch.cat((inputs.P, P), dim=0),
            V=torch.cat((inputs.V, V), dim=0),
            rays_base=cat_optional(inputs.rays_base, rays_base),
            rays_object=cat_optional(inputs.rays_object, rays_object),
            var_base=var_base,
            var_object=var_object,
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
        super().__init__(**kwargs)
        self.beam_diameter: Tensor = to_tensor(beam_diameter)

    def sample_light_source(
        self, sampling: dict[str, Any], dim: int, dtype: torch.dtype
    ) -> tuple[Tensor, Tensor, Tensor, Optional[Tensor]]:

        # Sample coordinates other than X on a disk
        NX = sampleND(
            sampling["base"],
            self.beam_diameter,
            dim - 1,
            dtype,
        )

        # Make the rays P + tV
        P = torch.column_stack((torch.zeros(NX.shape[0], dtype=dtype), NX))
        V = unit_vector(dim, dtype).unsqueeze(0)

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
            sampling["base"],
            self.beam_angular_size,
            dim - 1,
            dtype,
        )

        V = rotated_unit_vector(angles, dim)
        P = torch.zeros((1, dim), dtype=dtype)

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
            sampling["base"],
            self.beam_diameter,
            dim - 1,
            dtype,
        )
        P = torch.column_stack((torch.zeros(NX.shape[0], dtype=dtype), NX))

        # Sample angular direction
        angles = sampleND(
            sampling["object"],
            self.angular_size,
            dim - 1,
            dtype,
        )
        V = rotated_unit_vector(angles, dim)

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
            sampling["object"],
            self.object_diameter,
            dim - 1,
            dtype,
        )
        P = torch.column_stack((torch.zeros(NX.shape[0], dtype=dtype), NX))

        # Sample angular direction
        angles = sampleND(
            sampling["base"],
            self.beam_angular_size,
            dim - 1,
            dtype,
        )
        V = rotated_unit_vector(angles, dim)

        return P, V, angles, NX


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


class Wavelength(nn.Module):
    def __init__(self, lower: float | int, upper: float | int):
        super().__init__()
        self.lower, self.upper = lower, upper

    def forward(self, inputs: OpticalData) -> OpticalData:
        if inputs.rays_wavelength is not None:
            raise RuntimeError(
                "Rays already have wavelength data. Cannot apply Wavelength()."
            )

        if "wavelength" not in inputs.sampling:
            raise RuntimeError(
                "Missing 'wavelength' key in sampling configuration. Cannot apply Wavelength()."
            )

        chromatic_space = inputs.sampling["wavelength"].sample1d(self.lower, self.upper, inputs.dtype)

        return cartesian_wavelength(inputs, chromatic_space).replace(
            var_wavelength=chromatic_space
        )
