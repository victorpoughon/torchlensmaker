# This file is part of Torch Lens Maker
# Copyright (C) 2025 Victor Poughon
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

from typing import Any, Sequence, Optional, TypeAlias

from torchlensmaker.core.tensor_manip import to_tensor, filter_optional_tensor
from torchlensmaker.core.transforms import (
    TransformBase,
    TranslateTransform,
    LinearTransform,
    forward_kinematic,
    spherical_rotation,
)
from torchlensmaker.core.surfaces import (
    LocalSurface,
    CircularPlane,
)
from torchlensmaker.materials import (
    MaterialModel,
    NonDispersiveMaterial,
    get_material_model,
)
from torchlensmaker.core.physics import refraction, reflection, RefractionCriticalAngleMode
from torchlensmaker.core.intersect import intersect

from torchlensmaker.optical_data import OpticalData

Tensor = torch.Tensor

# Alias for convenience
Sequential: TypeAlias = nn.Sequential



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
        return inputs.replace(
            transforms=inputs.transforms + [self.kinematic_transform(dim, dtype)],
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

        return inputs.replace(loss=inputs.loss + loss)


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
    "Apply a rotation (in degrees) to the kinematic chain"

    def __init__(self, angles: tuple[float | int, float | int] | Tensor):
        super().__init__()

        if not isinstance(angles, torch.Tensor):
            angles = torch.as_tensor(angles, dtype=torch.float64)

        self.angles = angles

    def kinematic_transform(self, dim: int, dtype: torch.dtype) -> TransformBase:
        radangles = torch.deg2rad(self.angles)
        return spherical_rotation(radangles[0], radangles[1], dim, dtype)


class Offset(nn.Module):
    "Offset the given optical element but don't affect the kinematic chain"

    def __init__(self, element: nn.Module, x=0, y=0, z=0):
        super().__init__()
        self.element = element
        self.x = to_tensor(x)
        self.y = to_tensor(y)
        self.z = to_tensor(z)

    def forward(self, inputs: OpticalData) -> OpticalData:
        dim, dtype = inputs.dim, inputs.dtype

        if dim == 2:
            T = torch.stack((self.x, self.y), dim=0)
        else:
            T = torch.stack((self.x, self.y, self.z), dim=0)

        chain = inputs.transforms + [TranslateTransform(T)]

        # give that chain to the contained element
        element_input = inputs.replace(transforms=chain)

        element_output: OpticalData = self.element(element_input)

        # but return original transforms
        return element_output.replace(transforms=inputs.transforms)


class Rotate(nn.Module):
    "Rotate the given other optical element but don't affect the kinematic chain"

    # TODO should split 2D and 3D rotation
    # so that when a 3D rotation is used, 2D mode is properly undefined
    # instead of ignoring the second angle
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
        element_input = inputs.replace(transforms=chain)

        element_output: OpticalData = self.element(element_input)

        # but return original transforms
        return element_output.replace(transforms=inputs.transforms)


class KinematicSurface(nn.Module):
    """
    Position a child element according to surface anchors and scale
    Applies a different transform to the element itself and to the kinematic chain
    to support input and output anchors
    """

    def __init__(
        self,
        element: nn.Module,
        surface: LocalSurface,
        scale: float = 1.0,
        anchors: tuple[str, str] = ("origin", "origin"),
    ):
        super().__init__()
        self.element = element
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

        assert dtype == self.surface.dtype
        T = self.surface.extent(dim)

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

        assert dtype == self.surface.dtype
        S = self.scale * torch.eye(dim, dtype=dtype)
        S_inv = 1.0 / self.scale * torch.eye(dim, dtype=dtype)

        scale: Sequence[TransformBase] = [LinearTransform(S, S_inv)]

        extent_translate = -self.scale * self.surface.extent(dim)

        anchor: Sequence[TransformBase] = (
            [TranslateTransform(extent_translate)]
            if self.anchors[0] == "extent"
            else []
        )

        return list(anchor) + list(scale)

    def forward(self, inputs: OpticalData) -> OpticalData:
        dim, dtype = inputs.dim, inputs.dtype

        # give the surface transform to the contained element
        element_input = inputs.replace(
            transforms=inputs.transforms + self.surface_transform(dim, dtype)
        )

        element_output: OpticalData = self.element(element_input)

        # but return kinematic transform
        return element_output.replace(
            transforms=inputs.transforms + self.kinematic_transform(dim, dtype),
        )


class CollisionSurface(nn.Module):
    "Computes collisions and normals"

    def __init__(self, surface: LocalSurface):
        super().__init__()
        self.surface = surface

    def forward(self, inputs: OpticalData) -> OpticalData:
        dim, dtype = inputs.dim, inputs.dtype

        collision_points, surface_normals, valid = intersect(
            self.surface, inputs.P, inputs.V, forward_kinematic(inputs.transforms)
        )

        # Filter ray variables with valid collisions
        new_rays_base = filter_optional_tensor(inputs.rays_base, valid)
        new_rays_object = filter_optional_tensor(inputs.rays_object, valid)
        new_rays_wavelength = filter_optional_tensor(inputs.rays_wavelength, valid)

        return inputs.replace(
            P=collision_points,
            V=inputs.V[valid],
            normals=surface_normals,
            rays_base=new_rays_base,
            rays_object=new_rays_object,
            rays_wavelength=new_rays_wavelength,
            blocked=~valid,
        )


class ReflectiveBoundary(nn.Module):
    def forward(self, inputs: OpticalData) -> OpticalData:
        dim, dtype = inputs.dim, inputs.dtype

        if inputs.P.shape[0] == 0:
            return inputs

        if inputs.normals is None:
            raise RuntimeError(
                "Cannot compute ReflectiveBoundary without surface normals"
            )

        return inputs.replace(V=reflection(inputs.V, inputs.normals))


class RefractiveBoundary(nn.Module):
    def __init__(self, material, critical_angle):
        super().__init__()
        self.material = get_material_model(material)
        self.critical_angle = critical_angle

    def forward(self, inputs: OpticalData) -> OpticalData:
        dim, dtype = inputs.dim, inputs.dtype

        if inputs.P.shape[0] == 0:
            return inputs.replace(material=self.material)

        if inputs.normals is None:
            raise RuntimeError(
                "Cannot compute RefractiveBoundary without surface normals"
            )

        # if not torch.allclose(
        #     torch.linalg.vector_norm(inputs.normals, dim=1),
        #     torch.tensor(1.0, dtype=inputs.normals.dtype),
        # ):
        #     raise RuntimeError("surface normals should be unit vectors")

        if inputs.rays_wavelength is None:
            if not isinstance(inputs.material, NonDispersiveMaterial) or not isinstance(
                self.material, NonDispersiveMaterial
            ):
                raise RuntimeError(
                    f"Cannot compute refraction with dispersive material "
                    f"because optical data has no wavelength variable "
                    f"(got materials {inputs.material} and {self.material})"
                )

            n1 = inputs.material.n
            n2 = self.material.n

        else:
            n1 = inputs.material.refractive_index(inputs.rays_wavelength)
            n2 = self.material.refractive_index(inputs.rays_wavelength)

        refracted, valid = refraction(
            inputs.V, inputs.normals, n1, n2, critical_angle=self.critical_angle
        )

        if self.critical_angle == "drop":
            # 'drop' does the filtering internally
            # but still need to filter inputs
            new_P = inputs.P[valid]
            new_rays_base = filter_optional_tensor(inputs.rays_base, valid)
            new_rays_object = filter_optional_tensor(inputs.rays_object, valid)
            new_rays_wavelength = filter_optional_tensor(inputs.rays_wavelength, valid)
        else:
            new_P = inputs.P
            new_rays_base = inputs.rays_base
            new_rays_object = inputs.rays_object
            new_rays_wavelength = inputs.rays_wavelength

        return inputs.replace(
            P=new_P,
            V=refracted,
            normals=None,
            rays_base=new_rays_base,
            rays_object=new_rays_object,
            rays_wavelength=new_rays_wavelength,
            blocked=~valid,
            material=self.material,
        )


class RefractiveSurface(KinematicSurface):
    def __init__(
        self,
        surface: LocalSurface,
        material=str | MaterialModel,
        scale: float = 1.0,
        anchors: tuple[str, str] = ("origin", "origin"),
        critical_angle: RefractionCriticalAngleMode = "drop",
    ):
        element = nn.Sequential(
            CollisionSurface(surface),
            RefractiveBoundary(material, critical_angle),
        )
        super().__init__(element=element, surface=surface, scale=scale, anchors=anchors)


class ReflectiveSurface(KinematicSurface):
    def __init__(
        self,
        surface: LocalSurface,
        scale: float = 1.0,
        anchors: tuple[str, str] = ("origin", "origin"),
    ):
        element = nn.Sequential(
            CollisionSurface(surface),
            ReflectiveBoundary(),
        )
        super().__init__(element=element, surface=surface, scale=scale, anchors=anchors)


class Aperture(KinematicSurface):
    def __init__(self, diameter: float):
        surface = CircularPlane(diameter, dtype=torch.float64)  ## TODO dtype
        element = CollisionSurface(surface)
        super().__init__(
            element=element, surface=surface, scale=1.0, anchors=("origin", "origin")
        )


# TODO split into ImageBoundary and MagnificationLoss
class ImageBoundaryLoss(nn.Module):
    """
    Linear magnification circular image plane

    Loss function with a target magnification:
        L = (target_magnification - current_magnification)**2 + sum(residuals**2)

    Without:
        L = sum(residuals**2)
    """

    def __init__(
        self,
        magnification: Optional[int | float | Tensor] = None,
    ):
        super().__init__()
        self.magnification = (
            to_tensor(magnification) if magnification is not None else None
        )

    def forward(self, inputs: OpticalData) -> OpticalData:

        if inputs.V.shape[0] == 0:
            return inputs

        if inputs.normals is None:
            raise RuntimeError(
                "Cannot compute ImageBoundaryLoss data without surface normals"
            )

        if inputs.rays_object is None:
            raise RuntimeError(
                "Missing object coordinates on rays (required to compute image magnification)"
            )

        # Compute image surface coordinates here
        # To make this work with any surface, we would need a way to compute
        # surface coordinates for points on a surface, for any surface
        # For a plane it's easy though
        rays_image = inputs.P[:, 1:]
        rays_object = inputs.rays_object

        # Compute loss
        # could separate loss from imagesurface

        assert rays_object.shape == rays_image.shape
        mag, res = linear_magnification(rays_object, rays_image)

        if self.magnification is not None:
            loss = (self.magnification - mag) ** 2 + torch.sum(torch.pow(res, 2))
        else:
            loss = torch.sum(torch.pow(res, 2))

        return inputs.replace(
            rays_image=rays_image,
            loss=loss,
        )


class ImagePlane(KinematicSurface):
    def __init__(self, diameter, magnification=None):
        surface = CircularPlane(diameter, dtype=torch.float64)  ## TODO dtype
        element = nn.Sequential(
            CollisionSurface(surface),
            ImageBoundaryLoss(magnification),
        )
        super().__init__(
            element=element, surface=surface, scale=1.0, anchors=("origin", "origin")
        )
