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
    spherical_rotation,
)
from torchlensmaker.surfaces import (
    LocalSurface,
    CircularPlane,
)
from torchlensmaker.physics import refraction, reflection
from torchlensmaker.intersect import intersect
from torchlensmaker.sampling import (
    sample_line_linspace,
    sample_disk_linspace,
)


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

    # Tensor of dim 0
    # Loss accumulator
    loss: torch.Tensor

    def target(self) -> Tensor:
        dim, dtype = self.transforms[0].dim, self.transforms[0].dtype
        transform = forward_kinematic(self.transforms)
        return transform.direct_points(torch.zeros((dim,), dtype=dtype))


def default_input(sampling: dict[str, Any]) -> OpticalData:
    dim, dtype = sampling["dim"], sampling["dtype"]

    return OpticalData(
        sampling=sampling,
        transforms=[IdentityTransform(dim, dtype)],
        P=torch.empty((0, dim), dtype=dtype),
        V=torch.empty((0, dim), dtype=dtype),
        blocked=None,
        loss=torch.tensor(0.0, dtype=dtype),
    )


class FocalPoint(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs: OpticalData) -> OpticalData:
        dim = inputs.sampling["dim"]
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


class PointSourceAtInfinity(nn.Module):
    def __init__(self, beam_diameter: float):
        """
        Args:
            beam_diameter: diameter of the beam of light
            angle_offset: incidence angle of the beam (in degrees)
        """
        super().__init__()
        self.beam_diameter = beam_diameter

    def forward(self, inputs: OpticalData) -> OpticalData:
        dim, dtype = inputs.sampling["dim"], inputs.sampling["dtype"]

        N = inputs.sampling["base"]

        # Sample coordinates other than X
        if dim == 2:
            NX = sample_line_linspace(N, self.beam_diameter)
        else:
            # careful this does not sample exactly N points if N is not a perfect square
            NX = sample_disk_linspace(N, self.beam_diameter)

        # Make the rays P + tV
        P = torch.column_stack((torch.zeros(NX.shape[0], dtype=dtype), NX))
        unit_vect = torch.cat(
            (torch.ones(1, dtype=dtype), torch.zeros(dim - 1, dtype=dtype))
        )
        V = torch.tile(unit_vect, (P.shape[0], 1))

        # Make the rays 'base' coordinate
        base = NX

        # Apply kinematic transform
        tf = forward_kinematic(inputs.transforms)
        P = tf.direct_points(P)
        V = tf.direct_vectors(V)

        # Concatenate with existing rays
        # TODO some check that there are no other variables?
        return replace(
            inputs,
            P=torch.cat((inputs.P, P), dim=0),
            V=torch.cat((inputs.V, V), dim=0),
        )


class OpticalSurface(nn.Module):
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

    def surface_transform(self, dim: int, dtype: torch.dtype) -> list[TransformBase]:
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

    def chain_transform(self, dim: int, dtype: torch.dtype) -> Sequence[TransformBase]:
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

    def forward(self, inputs: OpticalData) -> OpticalData:
        assert inputs.P.shape[1] == inputs.V.shape[1] == inputs.sampling["dim"]

        dim, dtype = inputs.sampling["dim"], inputs.sampling["dtype"]

        surface_transform = forward_kinematic(
            inputs.transforms + self.surface_transform(dim, dtype)
        )

        collision_points, surface_normals, valid = intersect(
            self.surface, inputs.P, inputs.V, surface_transform
        )

        # Refract or reflect rays based on the derived class implementation
        output_rays = self.optical_function(
            inputs.V[valid],
            surface_normals,
        )

        chain_transform = self.chain_transform(dim, dtype)

        return replace(
            inputs,
            P=collision_points,
            V=output_rays,
            transforms=list(inputs.transforms) + list(chain_transform),
            blocked=~valid,
        )


class ReflectiveSurface(OpticalSurface):
    def __init__(
        self,
        surface: LocalSurface,
        scale: float = 1.0,
        anchors: tuple[str, str] = ("origin", "origin"),
    ):
        super().__init__(surface, scale, anchors)

    def optical_function(self, rays: Tensor, normals: Tensor) -> Tensor:
        return reflection(rays, normals)


class RefractiveSurface(OpticalSurface):
    def __init__(
        self,
        surface: LocalSurface,
        n: tuple[float, float],
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


class KinematicElement(nn.Module):
    """
    An element that appends a transform to the kinematic chain
    """

    def kinematic_transform(self, dim: int, dtype: torch.dtype) -> TransformBase:
        "Transform that gets appended to the kinematic chain by this element"
        raise NotImplementedError

    def forward(self, inputs: OpticalData) -> OpticalData:
        dim, dtype = inputs.sampling["dim"], inputs.sampling["dtype"]
        return replace(
            inputs,
            transforms=inputs.transforms + [self.kinematic_transform(dim, dtype)],
        )


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


class Rotate(nn.Module):
    "Rotate the given other optical element but don't affect the kinematic chain"

    def __init__(self, element: nn.Module, angle1: float, angle2: float):
        super().__init__()
        self.element = element
        self.angle1 = torch.deg2rad(torch.as_tensor(angle1, dtype=torch.float64))
        self.angle2 = torch.deg2rad(torch.as_tensor(angle2, dtype=torch.float64))

    def forward(self, inputs: OpticalData) -> OpticalData:
        dim, dtype = inputs.sampling["dim"], inputs.sampling["dtype"]

        chain = inputs.transforms + [
            spherical_rotation(self.angle1, self.angle2, dim, dtype)
        ]

        # give that chain to the contained element
        element_input = replace(inputs, transforms=chain)

        element_output: OpticalData = self.element(element_input)

        # but return original transforms
        return replace(element_output, transforms=inputs.transforms)
