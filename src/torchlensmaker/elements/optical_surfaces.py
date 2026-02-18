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

import torch
import torch.nn as nn

from typing import Sequence, Optional, TypeAlias, Literal, Self

from torchlensmaker.core.tensor_manip import to_tensor, filter_optional_tensor
from torchlensmaker.kinematics.homogeneous_geometry import (
    kinematic_chain_extend,
    kinematic_chain_append,
    hom_translate,
    hom_identity,
    HomMatrix,
    hom_scale,
)
from torchlensmaker.surfaces.local_surface import LocalSurface
from torchlensmaker.surfaces.plane import CircularPlane
from torchlensmaker.materials.get_material_model import (
    MaterialModel,
    get_material_model,
)
from torchlensmaker.physics.physics import (
    refraction,
    reflection,
)
from torchlensmaker.optical_data import OpticalData
from torchlensmaker.elements.sequential import SequentialElement
from torchlensmaker.elements.collision_surface import CollisionSurface

Tensor = torch.Tensor


AnchorType: TypeAlias = Literal["origin", "extent"]
MissMode: TypeAlias = Literal["absorb", "pass", "error"]


class ReflectiveSurface(SequentialElement):
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

    def reverse(self) -> Self:
        # TODO make a copy, surface should be a module
        return self

    def forward(self, data: OpticalData) -> OpticalData:
        t, normals, valid, fk_next = self.collision_surface(data)

        # full frame collision points
        collision_points = data.P + t.unsqueeze(1).expand_as(data.V) * data.V

        # Compute reflection for colliding rays
        reflected = reflection(data.V[valid], normals[valid])

        if self._miss == "absorb":
            # return hits only
            return data.filter_variables(valid).replace(
                P=collision_points[valid], V=reflected, fk=fk_next
            )

        elif self._miss == "pass":
            # insert hit rays as reflected
            return data.replace(
                P=data.P.masked_scatter(valid.unsqueeze(-1), collision_points[valid]),
                V=data.V.masked_scatter(valid.unsqueeze(-1), reflected),
                fk=fk_next,
            )

        elif self._miss == "error":
            misses = (~valid).sum()
            if misses != 0:
                raise RuntimeError(
                    f"Some rays ({misses}) don't collide with surface, but miss option is '{self._miss}'"
                )
            # return all rays as hits
            return data.replace(P=collision_points, V=reflected, fk=fk_next)


TIRMode: TypeAlias = Literal["absorb", "reflect"]


# TODO add miss mode support
# TODO support float material= as a shortcut for non-dispersive materials
class RefractiveSurface(SequentialElement):
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

    def reverse(self) -> Self:
        # TODO make a copy, surface should be a module
        return self

    def forward(self, data: OpticalData) -> tuple[OpticalData, Tensor]:
        # Collision detection
        t, normals, valid_collision, tf_next = self.collision_surface(data)
        collision_points = data.P + t.unsqueeze(-1).expand_as(data.V) * data.V

        # Zero rays special case
        # (needs to happen after self.collision_surface is called to enable rendering of it)
        if data.P.numel() == 0:
            return data.replace(fk=tf_next), torch.full((data.P.shape[0],), True)

        # Compute indices of refraction
        n1 = data.rays_index
        n2 = self.material(data.rays_wavelength)
        assert n1.shape == n2.shape == (data.P.shape[0],), (
            n1.shape,
            n2.shape,
            data.P.shape,
        )

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
            new_rays_pupil = filter_optional_tensor(data.rays_pupil, both_valid)
            new_rays_field = filter_optional_tensor(data.rays_field, both_valid)
            new_rays_wavelength = filter_optional_tensor(
                data.rays_wavelength, both_valid
            )
            new_rays_index = filter_optional_tensor(n2, both_valid)
        else:
            # keep tir rays
            new_P = collision_points[valid_collision]
            new_V = refracted[valid_collision]
            new_rays_pupil = filter_optional_tensor(data.rays_pupil, valid_collision)
            new_rays_field = filter_optional_tensor(data.rays_field, valid_collision)
            new_rays_wavelength = filter_optional_tensor(
                data.rays_wavelength, valid_collision
            )
            new_rays_index = filter_optional_tensor(n2, valid_collision)

        return data.replace(
            P=new_P,
            V=new_V,
            rays_pupil=new_rays_pupil,
            rays_field=new_rays_field,
            rays_wavelength=new_rays_wavelength,
            rays_index=new_rays_index,
            fk=tf_next,
        ), valid_refraction

    def sequential(self, data: OpticalData) -> OpticalData:
        output, _ = self(data)
        return output


class Aperture(SequentialElement):
    def __init__(self, diameter: float, dtype: torch.dtype | None = None):
        if dtype is None:
            dtype = torch.get_default_dtype()
        super().__init__()
        surface = CircularPlane(diameter, dtype=dtype)
        self.collision_surface = CollisionSurface(surface)

    def forward(self, data: OpticalData) -> OpticalData:
        # Collision detection
        t, _, valid_collision, tf_next = self.collision_surface(data)
        collision_points = data.P + t.unsqueeze(-1).expand_as(data.V) * data.V

        # Keep colliding rays only
        return data.replace(
            P=collision_points[valid_collision],
            V=data.V[valid_collision],
            rays_pupil=filter_optional_tensor(data.rays_pupil, valid_collision),
            rays_field=filter_optional_tensor(data.rays_field, valid_collision),
            rays_wavelength=filter_optional_tensor(
                data.rays_wavelength, valid_collision
            ),
            rays_index=filter_optional_tensor(data.rays_index, valid_collision),
            fk=tf_next,  # correct but useless cause Aperture is only circular plane currently
        )

    def reverse(self) -> Self:
        # TODO make a copy, surface should be a module
        return self


def linear_magnification(
    object_coordinates: Tensor, image_coordinates: Tensor
) -> tuple[Tensor, Tensor]:
    T, V = object_coordinates, image_coordinates

    # Fit linear magnification with least square and compute residuals
    mag = torch.sum(T * V) / torch.sum(T**2)
    residuals = V - mag * T

    return mag, residuals


class ImagePlane(SequentialElement):
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
        t, _, valid_collision, _ = self.collision_surface(data)
        collision_points = data.P + t.unsqueeze(-1).expand_as(data.V) * data.V

        if data.rays_field is None:
            raise RuntimeError(
                "Missing object coordinates on rays (required to compute image magnification)"
            )

        # TODO 2D only for now
        if data.dim == 3:
            return data

        # Compute image surface coordinates here
        # To make this work with any surface, we would need a way to compute
        # surface coordinates for points on a surface, for any surface
        # For a plane it's easy though

        rays_image = collision_points[:, 1]
        rays_object = data.rays_field

        # Compute loss
        assert rays_object.shape == rays_image.shape, (
            rays_object.shape,
            rays_image.shape,
        )
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
