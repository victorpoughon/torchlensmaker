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

from typing import Sequence, Optional, TypeAlias, Literal, cast

from torchlensmaker.core.tensor_manip import to_tensor, filter_optional_tensor
from torchlensmaker.core.transforms import (
    TransformBase,
    TranslateTransform,
    LinearTransform,
    forward_kinematic,
)
from torchlensmaker.surfaces.local_surface import LocalSurface
from torchlensmaker.surfaces.plane import CircularPlane
from torchlensmaker.materials import (
    MaterialModel,
    NonDispersiveMaterial,
    get_material_model,
)
from torchlensmaker.core.physics import (
    refraction,
    reflection,
)
from torchlensmaker.core.intersect import intersect

from torchlensmaker.optical_data import OpticalData

Tensor = torch.Tensor


class CollisionSurface(nn.Module):
    """
    Surface rays collision detection

    CollisionSurface implements collision detection between a bundle of rays
    (represented by an OpticalData object) and a surface. It also positions the
    surface according to anchors and scale options.

    Returns:
        t: full frame collision distances
        normals: full frame normals at the collision points
        valid: mask indicating which input rays collide with the surface
        chain: kinematic chain for the next element
    """

    def __init__(
        self,
        surface: LocalSurface,
        scale: float = 1.0,
        anchors: tuple[str, str] = ("origin", "origin"),
    ):
        super().__init__()
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

        # Subtract first anchor, add second anchor
        anchor0 = (
            [TranslateTransform(-self.scale * self.surface.extent(dim))]
            if self.anchors[0] == "extent"
            else []
        )
        anchor1 = (
            [TranslateTransform(self.scale * self.surface.extent(dim))]
            if self.anchors[1] == "extent"
            else []
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

        anchor: Sequence[TransformBase] = (
            [TranslateTransform(-self.scale * self.surface.extent(dim))]
            if self.anchors[0] == "extent"
            else []
        )

        return list(anchor) + list(scale)

    def forward(
        self, inputs: OpticalData
    ) -> tuple[Tensor, Tensor, Tensor, Sequence[TransformBase]]:
        dim, dtype = inputs.dim, inputs.dtype

        # Collision detection with the surface
        surface_tf = forward_kinematic(
            inputs.transforms + list(self.surface_transform(dim, dtype))
        )
        t, collision_normals, collision_valid = intersect(
            self.surface,
            inputs.P,
            inputs.V,
            surface_tf,
        )

        new_chain = inputs.transforms + list(self.kinematic_transform(dim, dtype))
        return t, collision_normals, collision_valid, new_chain


AnchorType: TypeAlias = Literal["origin", "extent"]
MissMode: TypeAlias = Literal["absorb", "pass", "error"]


class ReflectiveSurface(nn.Module):
    def __init__(
        self,
        surface: LocalSurface,
        scale: float = 1.0,
        anchors: tuple[AnchorType, AnchorType] = ("origin", "origin"),
        miss: MissMode = "absorb",
    ):
        super().__init__()
        self.collision_surface = CollisionSurface(surface, scale, anchors)
        self._miss = miss

    @property
    def surface(self) -> LocalSurface:
        return self.collision_surface.surface

    def forward(self, data: OpticalData) -> OpticalData:
        t, normals, valid, new_chain = self.collision_surface(data)

        # full frame collision points
        collision_points = data.P + t.unsqueeze(1).expand_as(data.V) * data.V

        # Compute reflection for colliding rays
        reflected = reflection(data.V[valid], normals[valid])

        if self._miss == "absorb":
            # return hits only
            return data.filter_variables(valid).replace(
                P=collision_points[valid],
                V=reflected,
                transforms=new_chain,
            )

        elif self._miss == "pass":
            # insert hit rays as reflected
            return data.replace(
                P=data.P.masked_scatter(valid.unsqueeze(-1), collision_points[valid]),
                V=data.V.masked_scatter(valid.unsqueeze(-1), reflected),
                transforms=new_chain,
            )

        elif self._miss == "error":
            misses = (~valid).sum()
            if misses != 0:
                raise RuntimeError(
                    f"Some rays ({misses}) don't collide with surface, but miss option is '{self._miss}'"
                )
            # return all rays as hits
            return data.replace(
                P=collision_points,
                V=reflected,
                transforms=new_chain,
            )


TIRMode: TypeAlias = Literal["absorb", "reflect"]


# TODO add miss mode support
class RefractiveSurface(nn.Module):
    def __init__(
        self,
        surface: LocalSurface,
        material: str | MaterialModel,
        scale: float = 1.0,
        anchors: tuple[str, str] = ("origin", "origin"),
        tir: TIRMode = "absorb",
        miss: MissMode = "absorb",
    ):
        super().__init__()
        self.collision_surface = CollisionSurface(surface, scale, anchors)
        self.material = get_material_model(material)
        self._miss = miss
        self._tir = tir

    @property
    def surface(self) -> LocalSurface:
        return self.collision_surface.surface

    def forward(self, data: OpticalData) -> tuple[OpticalData, Tensor]:
        # Collision detection
        t, normals, valid_collision, new_chain = self.collision_surface(data)
        collision_points = data.P + t.unsqueeze(-1).expand_as(data.V) * data.V

        # Zero rays special case
        # (needs to happen after self.collision_surface is called to enable rendering of it)
        if data.P.numel() == 0:
            return data.replace(material=self.material), torch.full(
                (data.P.shape[0],), True
            )

        # Compute indices of refraction (scalars for non dispersive materials,
        # tensors for dispersive materials)
        if (
            data.rays_wavelength is None
        ):  # TODO this should be empty but not None when number of rays is zero
            if not isinstance(data.material, NonDispersiveMaterial) or not isinstance(
                self.material, NonDispersiveMaterial
            ):
                raise RuntimeError(
                    f"Cannot compute refraction with dispersive material "
                    f"because optical data has no wavelength variable "
                    f"(got materials {data.material} and {self.material})"
                )

            n1 = torch.as_tensor(data.material.n)
            n2 = torch.as_tensor(self.material.n)

        else:
            n1 = data.material.refractive_index(data.rays_wavelength)
            n2 = self.material.refractive_index(data.rays_wavelength)

        # Snell's law happens here
        # Compute refraction on the full frame rays (including non-colliding
        # rays), so that comparing the two valid masks is easier
        refracted, valid_refraction = refraction(
            data.V, normals, n1, n2, critical_angle="reflect"
        )

        # noseq mode: rays, valid_tir

        both_valid = torch.logical_and(valid_collision, valid_refraction)

        if self._tir == "absorb":
            # filter TIR rays
            new_P = collision_points[both_valid]
            new_V = refracted[both_valid]
            new_rays_base = filter_optional_tensor(data.rays_base, both_valid)
            new_rays_object = filter_optional_tensor(data.rays_object, both_valid)
            new_rays_wavelength = filter_optional_tensor(
                data.rays_wavelength, both_valid
            )
        else:
            # keep tir rays
            new_P = collision_points[valid_collision]
            new_V = refracted[valid_collision]
            new_rays_base = filter_optional_tensor(data.rays_base, valid_collision)
            new_rays_object = filter_optional_tensor(data.rays_object, valid_collision)
            new_rays_wavelength = filter_optional_tensor(
                data.rays_wavelength, valid_collision
            )

        return data.replace(
            P=new_P,
            V=new_V,
            rays_base=new_rays_base,
            rays_object=new_rays_object,
            rays_wavelength=new_rays_wavelength,
            material=self.material,
            transforms=new_chain,
        ), valid_refraction

    def sequential(self, data: OpticalData) -> OpticalData:
        output, _ = self(data)
        return cast(OpticalData, output)


class Aperture(nn.Module):
    def __init__(self, diameter: float):
        super().__init__()
        surface = CircularPlane(diameter, dtype=torch.float64)  ## TODO dtype
        self.collision_surface = CollisionSurface(surface)

    def forward(self, data: OpticalData) -> OpticalData:
        # Collision detection
        t, _, valid_collision, new_chain = self.collision_surface(data)
        collision_points = data.P + t.unsqueeze(-1).expand_as(data.V) * data.V

        # Keep colliding rays only
        return data.replace(
            P=collision_points[valid_collision],
            V=data.V[valid_collision],
            rays_base=filter_optional_tensor(data.rays_base, valid_collision),
            rays_object=filter_optional_tensor(data.rays_object, valid_collision),
            rays_wavelength=filter_optional_tensor(
                data.rays_wavelength, valid_collision
            ),
            transforms=new_chain,  # correct but useless cause Aperture is only circular plane currently
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


def linear_magnification(
    object_coordinates: Tensor, image_coordinates: Tensor
) -> tuple[Tensor, Tensor]:
    T, V = object_coordinates, image_coordinates

    # Fit linear magnification with least square and compute residuals
    mag = torch.sum(T * V) / torch.sum(T**2)
    residuals = V - mag * T

    return mag, residuals


class ImagePlane(nn.Module):
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
    ):
        super().__init__()
        self.collision_surface = CollisionSurface(CircularPlane(diameter))
        self.magnification = (
            to_tensor(magnification) if magnification is not None else None
        )

    def forward(self, data: OpticalData) -> OpticalData:
        if data.V.shape[0] == 0:
            return data

        # Collision detection
        t, _, valid_collision, new_chain = self.collision_surface(data)
        collision_points = data.P + t.unsqueeze(-1).expand_as(data.V) * data.V

        if data.rays_object is None:
            raise RuntimeError(
                "Missing object coordinates on rays (required to compute image magnification)"
            )

        # Compute image surface coordinates here
        # To make this work with any surface, we would need a way to compute
        # surface coordinates for points on a surface, for any surface
        # For a plane it's easy though
        # TODO 2D only for now
        # assert data.dim == 2 # assert disabled so show3d works
        rays_image = collision_points[:, 1:]
        rays_object = data.rays_object

        # Compute loss

        assert rays_object.shape == rays_image.shape
        mag, res = linear_magnification(rays_object, rays_image)

        if self.magnification is not None:
            loss = (self.magnification - mag) ** 2 + torch.sum(torch.pow(res, 2))
        else:
            loss = torch.sum(torch.pow(res, 2))

        # what to do with rays after an image plane?

        return data.replace(
            rays_image=rays_image,  # used by spot_diagram
            loss=loss,
        )
