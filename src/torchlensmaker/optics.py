import torch
import torch.nn as nn
from dataclasses import dataclass, replace

from typing import Any, Sequence, Optional

from torchlensmaker.tensor_manip import to_tensor
from torchlensmaker.transforms import (
    TransformBase,
    TranslateTransform,
    LinearTransform,
    IdentityTransform,
    forward_kinematic,
    spherical_rotation,
)
from torchlensmaker.surfaces import (
    LocalSurface,
    CircularPlane,
)
from torchlensmaker.physics import refraction, reflection
from torchlensmaker.intersect import intersect


Tensor = torch.Tensor

# Alias for convenience
Sequential = nn.Sequential


@dataclass
class OpticalData:
    # dim is 2 or 3
    # dtype default is torch.float64
    dim: int
    dtype: torch.dtype

    # Sampling configuration
    sampling: dict[str, Any]

    # Transform kinematic chain
    transforms: list[TransformBase]

    # Parametric light rays P + tV
    # Tensors of shape (N, 2|3)
    P: Tensor
    V: Tensor

    # Rays variables
    # Tensors of shape (N, 2|3) or None
    rays_base: Optional[Tensor]
    rays_object: Optional[Tensor]
    rays_image: Optional[Tensor]

    # Mask array indicating which rays from the previous data in the sequence
    # were blocked by the previous optical element. "blocked" includes hitting
    # an absorbing surface but also not hitting anything
    # None or Tensor of shape (N,)
    blocked: Optional[Tensor]

    # Loss accumulator
    # Tensor of dim 0
    loss: torch.Tensor

    def target(self) -> Tensor:
        dim, dtype = self.transforms[0].dim, self.transforms[0].dtype
        transform = forward_kinematic(self.transforms)
        return transform.direct_points(torch.zeros((dim,), dtype=dtype))

    def replace(self, /, **changes: Any) -> "OpticalData":
        return replace(self, **changes)


def default_input(
    dim: int, dtype: torch.dtype, sampling: dict[str, Any]
) -> OpticalData:
    return OpticalData(
        dim=dim,
        dtype=dtype,
        sampling=sampling,
        transforms=[IdentityTransform(dim, dtype)],
        P=torch.empty((0, dim), dtype=dtype),
        V=torch.empty((0, dim), dtype=dtype),
        rays_base=None,
        rays_object=None,
        rays_image=None,
        blocked=None,
        loss=torch.tensor(0.0, dtype=dtype),
    )


def linear_magnification(
    object_coordinates: Tensor, image_coordinates: Tensor
) -> tuple[Tensor, Tensor]:
    T, V = object_coordinates, image_coordinates

    # Fit linear magnification with least square and compute residuals
    mag = torch.sum(T * V) / torch.sum(T**2)
    residuals = V - mag * T

    return mag, residuals


class KinematicElement(nn.Module):
    """
    Skeleton element that appends a transform to the kinematic chain
    """

    def kinematic_transform(self, dim: int, dtype: torch.dtype) -> TransformBase:
        "Transform that gets appended to the kinematic chain by this element"
        raise NotImplementedError

    def forward(self, inputs: OpticalData) -> OpticalData:
        dim, dtype = inputs.dim, inputs.dtype
        return replace(
            inputs,
            transforms=inputs.transforms + [self.kinematic_transform(dim, dtype)],
        )


class SurfaceMixin:
    """
    Mixin to hold a reference to a scaled and anchored surface and automatically
    registers any of its parameters. Also provides the intersect_surface() function.
    """

    def __init__(
        self,
        surface: LocalSurface,
        scale: float = 1.0,
        anchors: tuple[str, str] = ("origin", "origin"),
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.surface = surface
        self.scale = scale
        self.anchors = anchors

        # If surface has parameters, register them
        for name, p in surface.parameters().items():
            self.register_parameter(name, p)

    def kinematic_transform(
        self, dim: int, dtype: torch.dtype
    ) -> Sequence[TransformBase]:
        "Additional transform that applies to the next element"

        T = self.surface.extent(dim, dtype)

        # Subtract first anchor, add second anchor
        anchor0 = (
            [TranslateTransform(-self.scale * T)] if self.anchors[0] == "extent" else []
        )
        anchor1 = (
            [TranslateTransform(self.scale * T)] if self.anchors[1] == "extent" else []
        )

        return list(anchor0) + list(anchor1)

    def surface_transform(
        self, dim: int, dtype: torch.dtype
    ) -> Sequence[TransformBase]:
        "Additional transform that applies to the surface"

        S = self.scale * torch.eye(dim, dtype=dtype)
        S_inv = 1.0 / self.scale * torch.eye(dim, dtype=dtype)

        scale: Sequence[TransformBase] = [LinearTransform(S, S_inv)]

        extent_translate = -self.scale * self.surface.extent(dim, dtype)

        anchor: Sequence[TransformBase] = (
            [TranslateTransform(extent_translate)]
            if self.anchors[0] == "extent"
            else []
        )

        return list(anchor) + list(scale)

    def intersect_surface(self, inputs: OpticalData) -> tuple[Tensor, Tensor, Tensor]:
        dim, dtype = inputs.dim, inputs.dtype

        surface_transform = forward_kinematic(
            inputs.transforms + list(self.surface_transform(dim, dtype))
        )

        return intersect(self.surface, inputs.P, inputs.V, surface_transform)


class ImagePlane(SurfaceMixin, nn.Module):
    """
    Linear magnification circular image plane

    Loss function with a target magnification:
        L = (target_magnification - current_magnification)**2 + sum(residuals**2)

    Without:
        L = sum(residuals**2)
    """

    def __init__(
        self,
        diameter: float,
        magnification: Optional[int | float | Tensor] = None,
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__(surface=CircularPlane(diameter, dtype))
        self.magnification = (
            to_tensor(magnification) if magnification is not None else None
        )

    def forward(self, inputs: OpticalData) -> OpticalData:

        if inputs.V.shape[0] == 0:
            return inputs

        # Intersect with the image surface
        collision_points, surface_normals, valid = self.intersect_surface(inputs)

        # Filter ray variables with valid collisions
        valid_rays_base = filter_optional_tensor(inputs.rays_base, valid)
        valid_rays_object = filter_optional_tensor(inputs.rays_object, valid)

        # Compute image surface coordinates here
        # To make this work with any surface, we would need a way to compute
        # surface coordinates for points on a surface, for any surface
        # For a plane it's easy though
        rays_image = collision_points[:, 1:]
        rays_object = valid_rays_object

        if rays_object is None:
            raise RuntimeError("Missing object coordinates on rays (required to compute image magnification)")

        # Compute loss
        # could separate loss from imagesurface

        assert rays_object.shape == rays_image.shape, (rays_object.shape, rays_image.shape)
        mag, res = linear_magnification(rays_object, rays_image)

        if self.magnification is not None:
            loss = (self.magnification - mag) ** 2 + torch.sum(torch.pow(res, 2))
        else:
            loss = torch.sum(torch.pow(res, 2))
        
        return replace(
            inputs,
            P=collision_points,
            V=inputs.V[valid],
            rays_base=valid_rays_base,
            rays_object=valid_rays_object,
            rays_image=rays_image,
            loss=loss,
            blocked=~valid,
        )


class FocalPoint(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs: OpticalData) -> OpticalData:
        dim = inputs.dim
        N = inputs.P.shape[0]

        X = inputs.target()
        P = inputs.P
        V = inputs.V

        # Compute ray-point squared distance distance

        # If 2D, pad to 3D with zeros
        if dim == 2:
            X = torch.cat((X, torch.zeros(1)), dim=0)
            P = torch.cat((P, torch.zeros((N, 1))), dim=1)
            V = torch.cat((V, torch.zeros((N, 1))), dim=1)

        cross = torch.cross(X - P, V, dim=1)
        norm = torch.norm(V, dim=1)

        distance = torch.norm(cross, dim=1) / norm

        loss = distance.sum() / N

        return replace(inputs, loss=inputs.loss + loss)


class OpticalSurface(SurfaceMixin, nn.Module):
    "Skeleton element for a kinematic and optical surface"

    def forward(self, inputs: OpticalData) -> OpticalData:
        dim, dtype = inputs.dim, inputs.dtype

        collision_points, surface_normals, valid = self.intersect_surface(inputs)

        # Refract or reflect rays based on the derived class implementation
        output_rays = self.optical_function(
            inputs.V[valid],
            surface_normals,
        )

        # Filter ray variables with valid collisions
        new_rays_base = filter_optional_tensor(inputs.rays_base, valid)
        new_rays_object = filter_optional_tensor(inputs.rays_object, valid)

        # Apply surface kinematic transform to the chain
        new_transforms = inputs.transforms + list(self.kinematic_transform(dim, dtype))

        return replace(
            inputs,
            P=collision_points,
            V=output_rays,
            rays_base=new_rays_base,
            rays_object=new_rays_object,
            transforms=new_transforms,
            blocked=~valid,
        )


def filter_optional_tensor(t: Optional[Tensor], valid: Tensor) -> Optional[Tensor]:
    if t is None:
        return None
    else:
        return t[valid]


class ReflectiveSurface(OpticalSurface):
    def optical_function(self, rays: Tensor, normals: Tensor) -> Tensor:
        return reflection(rays, normals)


class RefractiveSurface(OpticalSurface):
    def __init__(
        self,
        n: tuple[float, float],
        surface: LocalSurface,
        scale: float = 1.0,
        anchors: tuple[str, str] = ("origin", "origin"),
    ):
        super().__init__(surface, scale, anchors)
        self.n1, self.n2 = n

    def optical_function(self, rays: Tensor, normals: Tensor) -> Tensor:
        return refraction(rays, normals, self.n1, self.n2, critical_angle="clamp")


class Aperture(OpticalSurface):
    def __init__(self, diameter: float):
        surface = CircularPlane(diameter, dtype=torch.float64)
        super().__init__(surface, 1.0, ("origin", "origin"))

    def optical_function(self, rays: Tensor, _normals: Tensor) -> Tensor:
        return rays


class Gap(KinematicElement):
    def __init__(self, offset: float | int | Tensor):
        super().__init__()
        assert isinstance(offset, (float, int, torch.Tensor))
        if isinstance(offset, torch.Tensor):
            assert offset.dim() == 0

        # Gap is always stored as float64, but it's converted to the sampling
        # dtype when creating the corresponding transform in forward()
        self.offset = torch.as_tensor(offset, dtype=torch.float64)

    def kinematic_transform(self, dim: int, dtype: torch.dtype) -> TransformBase:
        translate_vector = torch.cat(
            (
                self.offset.unsqueeze(0).to(dtype=dtype),
                torch.zeros(dim - 1, dtype=dtype),
            )
        )

        return TranslateTransform(translate_vector)


class Turn(KinematicElement):
    def __init__(self, angles: tuple[float | int, float | int] | Tensor):
        super().__init__()

        if not isinstance(angles, torch.Tensor):
            angles = torch.as_tensor(angles, dtype=torch.float64)

        self.angles = torch.deg2rad(angles)

    def kinematic_transform(self, dim: int, dtype: torch.dtype) -> TransformBase:
        return spherical_rotation(self.angles[0], self.angles[1], dim, dtype)


class Rotate(nn.Module):
    "Rotate the given other optical element but don't affect the kinematic chain"

    def __init__(
        self, element: nn.Module, angles: tuple[float | int, float | int] | Tensor
    ):
        super().__init__()
        self.element = element
        if not isinstance(angles, torch.Tensor):
            self.angles = torch.deg2rad(torch.as_tensor(angles, dtype=torch.float64))

    def forward(self, inputs: OpticalData) -> OpticalData:
        dim, dtype = inputs.dim, inputs.dtype

        chain = inputs.transforms + [
            spherical_rotation(self.angles[0], self.angles[1], dim, dtype)
        ]

        # give that chain to the contained element
        element_input = replace(inputs, transforms=chain)

        element_output: OpticalData = self.element(element_input)

        # but return original transforms
        return replace(element_output, transforms=inputs.transforms)
