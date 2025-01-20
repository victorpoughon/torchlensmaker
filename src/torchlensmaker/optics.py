import torch
import torch.nn as nn
from dataclasses import dataclass, replace

from typing import Any, Sequence, Optional

from torchlensmaker.tensorframe import TensorFrame
from torchlensmaker.transforms import (
    TransformBase,
    TranslateTransform,
    LinearTransform,
    IdentityTransform,
    forward_kinematic,
)
from torchlensmaker.physics import refraction, reflection
from torchlensmaker.rot2d import rot2d
from torchlensmaker.rot3d import euler_angles_to_matrix
from torchlensmaker.intersect import intersect


Tensor = torch.Tensor


@dataclass
class OpticalData:
    # sampling information
    sampling: dict[str, Any]

    # Transform kinematic chain
    transforms: list[TransformBase]

    # Tensors of shape (N, 2|3)
    # Parametric light rays P + tV
    P: Tensor
    V: Tensor

    # None or Tensor of shape (N,)
    # Mask array indicating which rays from the previous data in the sequence
    # were blocked by the previous optical element. "blocked" includes hitting
    # an absorbing surface but also not hitting anything
    blocked: Optional[Tensor]


def default_input(sampling: dict[str, Any]) -> OpticalData:
    dim, dtype = sampling["dim"], sampling["dtype"]

    return OpticalData(
        sampling=sampling,
        transforms=[IdentityTransform(dim, dtype)],
        P=torch.empty((0, dim)),
        V=torch.empty((0, dim)),
        blocked=None,
    )


# light sources have to be in the optical sequence
# but not necessarily on the kinematic chain


class PointSourceAtInfinity(nn.Module):
    """
    A simple light source that models a perfect point at infinity.

    All rays are parallel with possibly some incidence angle
    """

    def __init__(self, beam_diameter, angle1=0.0, angle2=0.0):
        """
        beam_diameter: diameter of the beam of parallel light rays
        angle1: angle of indidence with respect to the principal axis, in degrees
        angle2: second angle of incidence used in 3D

        samples along the base sampling dimension
        """

        super().__init__()
        self.beam_diameter = torch.as_tensor(beam_diameter, dtype=torch.float64)
        self.angle1 = torch.deg2rad(torch.as_tensor(angle1, dtype=torch.float64))
        self.angle2 = torch.deg2rad(torch.as_tensor(angle2, dtype=torch.float64))

    def forward(self, inputs):
        # Create new rays by sampling the beam diameter
        dim, dtype = inputs.sampling["dim"], inputs.sampling["dtype"]
        num_rays = inputs.sampling["base"]
        margin = 0.1  # TODO

        # rays origins
        D = self.beam_diameter
        RY = torch.linspace(-D / 2 + margin, D / 2 - margin, num_rays)

        if dim == 3:
            RZ = RY.clone()

        if dim == 2:
            RX = torch.zeros(num_rays)
            rays_origins = torch.column_stack((RX, RY))
        else:
            RX = torch.zeros(num_rays * num_rays)
            prod = torch.cartesian_prod(RY, RZ)
            rays_origins = torch.column_stack((RX, prod[:, 0], prod[:, 1]))

        # rays vectors
        if dim == 2:
            V = torch.tensor([1.0, 0.0], dtype=dtype)

            vect = rot2d(V, self.angle1)
        else:
            V = torch.tensor([1.0, 0.0, 0.0], dtype=dtype)
            M = euler_angles_to_matrix(
                torch.deg2rad(
                    torch.as_tensor([0.0, self.angle1, self.angle2], dtype=dtype)
                ),
                "ZYX",
            ).to(
                dtype=dtype
            )  # TODO need to support dtype in euler_angles_to_matrix
            vect = V @ M

        assert vect.dtype == dtype

        if dim == 2:
            rays_vectors = torch.tile(vect, (num_rays, 1))
        else:
            # use the same base dimension twice here
            # TODO could define different ones
            rays_vectors = torch.tile(vect, (num_rays * num_rays, 1))

        # transform sources to the chain target
        transform = forward_kinematic(inputs.transforms)
        rays_origins = transform.direct_points(rays_origins)
        rays_vectors = transform.direct_vectors(rays_vectors)

        # normalized coordinate along the base dimension
        # coord_base = (RY + self.beam_diameter / 2) / self.beam_diameter


        assert rays_origins.shape[1] == dim, rays_origins.shape
        assert rays_vectors.shape[1] == dim, rays_vectors.shape


        return replace(
            inputs,
            P=torch.cat((inputs.P, rays_origins), dim=0),
            V=torch.cat((inputs.V, rays_vectors), dim=0),
        )


class OpticalSurface(nn.Module):
    def __init__(self, surface, scale=1.0, anchors=("origin", "origin")):
        super().__init__()
        self.surface = surface
        self.scale = scale
        self.anchors = anchors

        # mode 1:  inline / chained / affecting
        # surface transform = input.transforms - anchor + scale
        # output transform = input.transforms - first anchor + second anchor

        # mode 2: offline / free / independent
        # surface transform = input.transforms + local transform - anchor + scale
        # output transform = input.transforms

        # how to support absolute position on chain?

        # RS(X - A) + T
        # surface transform(X) = CSX - A
        # surface transform = anchor1 + scale + chain
        # output transform = chain + anchor1 + anchor2

    def surface_transform(self, dim, dtype) -> Sequence[TransformBase]:
        "Transform chain that applies to the underlying surface"

        S = self.scale * torch.eye(dim, dtype=dtype)
        S_inv = 1.0 / self.scale * torch.eye(dim, dtype=dtype)

        scale: list[TransformBase] = [LinearTransform(S, S_inv)]
        anchor: list[TransformBase]

        if self.anchors[0] == "extent":
            T = -self.scale * torch.cat(
                (self.surface.extent().unsqueeze(0), torch.zeros(dim - 1, dtype=dtype)),
                dim=0,
            )
            anchor = [TranslateTransform(T)]
        else:
            anchor = []

        return anchor + scale

    def output_transform(self, dim, dtype) -> Sequence[TransformBase]:
        "Transform chain that applies to the next element"

        # subtract first anchor, add second anchor

        if self.anchors[0] == "extent":
            Ta = -self.scale * torch.cat(
                (self.surface.extent().unsqueeze(0), torch.zeros(dim - 1, dtype=dtype)),
                dim=0,
            )
            anchor1 = [TranslateTransform(Ta)]
        else:
            anchor1 = []

        if self.anchors[1] == "extent":
            Tb = self.scale * torch.cat(
                (self.surface.extent().unsqueeze(0), torch.zeros(dim - 1, dtype=dtype)),
                dim=0,
            )
            anchor2 = [TranslateTransform(Tb)]
        else:
            anchor2 = []

        return anchor1 + anchor2

    def forward(self, inputs):
        dim, dtype = inputs.sampling["dim"], inputs.sampling["dtype"]

        surface_transform = forward_kinematic(
            inputs.transforms + self.surface_transform(dim, dtype)
        )

        assert inputs.P.shape[1] == dim
        assert inputs.V.shape[1] == dim

        collision_points, surface_normals, valid = intersect(
            self.surface, inputs.P, inputs.V, surface_transform
        )

        # Keep only non blocked rays
        V_valid = inputs.V[valid]

        # Verify no weirdness in the data
        assert torch.all(torch.isfinite(collision_points))
        assert torch.all(torch.isfinite(surface_normals))

        # Refract or reflect rays based on the derived class implementation
        output_rays = refraction(
            V_valid, surface_normals, 1.0, 1.5, critical_angle="clamp"
        )

        output_transform = self.output_transform(dim, dtype)

        return replace(
            inputs,
            P = collision_points,
            V = output_rays,
            transforms=inputs.transforms + output_transform,
            blocked=~valid,
        )


class Gap(nn.Module):
    def __init__(self, offset: float | int | Tensor):
        super().__init__()
        assert isinstance(offset, (float, int, torch.Tensor))
        if isinstance(offset, torch.Tensor):
            assert offset.dim() == 0

        # Gap is always stored as float64, but it's converted to the sampling
        # dtype when creating the corresponding transform in forward()
        self.offset = torch.as_tensor(offset, dtype=torch.float64)

    def forward(self, inputs):
        dim, dtype = inputs.sampling["dim"], inputs.sampling["dtype"]

        translate_vector = torch.cat(
            (
                self.offset.unsqueeze(0).to(dtype=dtype),
                torch.zeros(dim - 1, dtype=dtype),
            )
        )

        return replace(
            inputs,
            transforms=inputs.transforms + [TranslateTransform(translate_vector)],
        )
