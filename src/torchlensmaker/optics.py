import torch
import torch.nn as nn

from torchlensmaker.module import Module

from typing import Optional
from dataclasses import dataclass, replace

from torchlensmaker.raytracing import (
    refraction,
    reflection,
    ray_point_squared_distance,
    position_on_ray,
    rays_to_coefficients,
    rot2d,
)

from torchlensmaker.torch_extensions import OpticalSequence

from torchlensmaker.surface import Surface
from torchlensmaker.shapes import Line

from torchlensmaker.tensorframe import TensorFrame


def loss_nonpositive(parameters, scale=1):
    return torch.where(parameters > 0, torch.pow(scale*parameters, 2), torch.zeros_like(parameters))


@dataclass
class OpticalData:
    """
    Holder class for the data that's passed between optical elements
    """

    # TensorFrame of light rays
    rays: TensorFrame

    # Tensor of shape (2,)
    # Position of the next optical element
    target: torch.Tensor

    # None or Tensor of shape (N,)
    # Mask array indicating which rays from the previous data in the optical
    # stack were blocked by the previous optical element
    # "block" includes hitting an absorbing surface but also not hitting anything
    blocked: Optional[torch.Tensor]

    # Tensor of one element
    # Loss accumulator
    loss: torch.Tensor


default_input = OpticalData(
    rays = TensorFrame(torch.empty((0, 4)), columns = ["RX", "RY", "VX", "VY"]),
    target = torch.zeros(2),
    blocked = None,
    loss = torch.tensor(0.),
)


class FocalPoint(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs: OpticalData, sampling: dict):
        num_rays = inputs.rays.shape[0]
        rays_origins, rays_vectors = (
            inputs.rays.get(["RX", "RY"]),
            inputs.rays.get(["VX", "VY"]),
        )
        sum_squared = ray_point_squared_distance(rays_origins, rays_vectors, inputs.target).sum()
        loss = sum_squared / num_rays

        return replace(inputs, loss=inputs.loss + loss)


class Image(nn.Module):
    "An image is a set of focal points that map to the observed object"

    def __init__(self, height):
        super().__init__()
        self.height = torch.as_tensor(height, dtype=torch.float32)
    
    def forward(self, inputs: OpticalData, sampling: dict):
        # Compute image loss

        # First, make the 2D points that correspond to the object sampling
        coords_object = inputs.rays.get("object")
        rays_origins, rays_vectors = (
            inputs.rays.get(["RX", "RY"]),
            inputs.rays.get(["VX", "VY"]),
        )
        points_y = coords_object * self.height - self.height / 2
        points_x = inputs.target[0].expand_as(points_y)

        points = torch.stack((points_x, points_y), dim=-1)

        num_rays = rays_origins.shape[0]
        sum_squared = ray_point_squared_distance(rays_origins, rays_vectors, points).sum()
        loss = sum_squared / num_rays

        return replace(inputs, loss=inputs.loss + loss)


class ImagePlane(nn.Module):
    "An image is a set of focal points that map to the observed object"

    def __init__(self, height):
        super().__init__()
        self.height = torch.as_tensor(height, dtype=torch.float32) # TODO needs a height only for rendering
    
    def forward(self, inputs: OpticalData, sampling: dict):
        # Compute image coordinates of rays hitting the image plane

        # image plane coordinates
        orig, V = (
            inputs.rays.get(["RX", "RY"]),
            inputs.rays.get(["VX", "VY"]),
        )
        a, b, c = -V[:, 1], V[:, 0], V[:, 1] * orig[:, 0] - V[:, 0] * orig[:, 1]
        X = torch.full_like(a, inputs.target[0].item())
        Y = (- c - a*X ) / b

        # Add the image coordinate column to the rays TensorFrame
        return replace(
            inputs,
            rays=inputs.rays.update(image=Y)
        )


class PointSource(nn.Module):
    def __init__(self, beam_angle, height=0, object_coord=0.):
        """
        height: height of the point source above the principal axis
        beam_angle: total angle of the emitted beam of rays (in degrees)
        """

        super().__init__()
        self.beam_angle = torch.deg2rad(
            torch.as_tensor(beam_angle, dtype=torch.float32)
        )
        self.height = torch.as_tensor(height, dtype=torch.float32)

        # TODO remove this and do sampling directly in object?
        self.object_coord = torch.as_tensor(object_coord, dtype=torch.float32)

    def forward(self, inputs: OpticalData, sampling: dict):

        num_rays = sampling["rays"]

        # Create new rays by sampling the beam angle
        rays_origins = torch.tile(
            inputs.target + torch.tensor([0.0, self.height]), (num_rays, 1)
        )

        angles = torch.linspace(
            -self.beam_angle / 2, self.beam_angle / 2, num_rays
        )
        rays_vectors = rot2d(torch.tensor([1.0, 0.0]), angles)

        # normalized coordinate along the base dimension
        coord_base = (angles + self.beam_angle / 2) / self.beam_angle
        coord_object = self.object_coord.expand_as(coord_base)

        new_rays = TensorFrame(
            torch.cat((rays_origins, rays_vectors, coord_base.unsqueeze(1), coord_object.unsqueeze(1)), dim=1),
            columns=["RX", "RY", "VX", "VY", "rays", "object"],
        )

        # Add new rays to the input rays
        return OpticalData(
            inputs.rays.stack(new_rays),
            inputs.target,
            None,
            inputs.loss,
        )


class PointSourceAtInfinity(nn.Module):
    def __init__(self, beam_diameter, angle=0., object_coord=0.):
        """
        beam_diameter: diameter of the beam of parallel light rays
        angle: angle of indidence with respect to the principal axis, in degrees

        samples along the base sampling dimension
        """

        super().__init__()
        self.beam_diameter = torch.as_tensor(beam_diameter, dtype=torch.float32)
        self.angle = torch.deg2rad(torch.as_tensor(angle, dtype=torch.float32))
        self.object_coord = torch.as_tensor(object_coord, dtype=torch.float32)

    def forward(self, inputs: OpticalData, sampling: dict):
        # Create new rays by sampling the beam diameter
        num_rays = sampling["rays"]
        margin = 0.1  # TODO
        RX = torch.zeros(num_rays)
        RY = torch.linspace(
            -self.beam_diameter / 2 + margin,
            self.beam_diameter / 2 - margin,
            num_rays,
        )

        rays_origins = inputs.target + torch.column_stack((RX, RY))
        vect = rot2d(torch.tensor([1.0, 0.0]), self.angle)
        rays_vectors = torch.tile(vect, (num_rays, 1))

        # normalized coordinate along the base dimension
        coord_base = (RY + self.beam_diameter / 2) / self.beam_diameter

        coord_object = self.object_coord.expand_as(coord_base)

        new_rays = TensorFrame(
            torch.cat((rays_origins, rays_vectors, coord_base.unsqueeze(1), coord_object.unsqueeze(1)), dim=1),
            columns=["RX", "RY", "VX", "VY", "rays", "object"],
        )

        return OpticalData(
            inputs.rays.stack(new_rays),
            inputs.target,
            None,
            inputs.loss,
        )


class ObjectAtInfinity(nn.Module):
    def __init__(self, beam_diameter, angular_size, angle=0):
        """
        angular_size: apparent angular size of the object, in degrees
        angle: angle of incidence of the object's center with the principal axis, in degrees
        """

        super().__init__()
        self.beam_diameter = torch.as_tensor(beam_diameter, dtype=torch.float32)
        self.angular_size = torch.as_tensor(angular_size, dtype=torch.float32)
        self.angle = torch.deg2rad(torch.as_tensor(angle, dtype=torch.float32))

    def forward(self, inputs: OpticalData, sampling: dict):
        # An object at infinity is a collection of points at infinity,
        # sampled along the object's angular size

        num_samples = sampling["object"]

        angles = torch.linspace(-self.angular_size/2., self.angular_size/2, num_samples)

        modules = OpticalSequence()

        for angle in angles:
            # add a PointSourceAtInfinity with that object coordinate
            current_angle = angle
            mod = PointSourceAtInfinity(self.beam_diameter, angle=current_angle + self.angle, object_coord=current_angle)
            modules.append(mod)

        return modules.forward(inputs, sampling)


class Gap(nn.Module):
    def __init__(self, offset):
        super().__init__()
        self.offset = offset
    
    def forward(self, inputs: OpticalData, sampling: dict):
        offset = torch.stack((torch.as_tensor(self.offset), torch.tensor(0.)))
        new_target = inputs.target + offset

        return OpticalData(inputs.rays, new_target, None, inputs.loss)


class Aperture(nn.Module):
    def __init__(self, height, diameter):
        super().__init__()
        self.height = height
        self.diameter = diameter
        self.shape = Line(diameter)
    
    def forward(self, inputs: OpticalData, sampling: dict):
        surface = Surface(self.shape, pos=inputs.target)

        # TODO factor common collision code with OpticalSurface
        # For all rays, find the intersection with the surface and the normal vector at the intersection
        rays_origins, rays_vectors = (
            inputs.rays.get(["RX", "RY"]),
            inputs.rays.get(["VX", "VY"]),
        )
        lines = rays_to_coefficients(rays_origins, rays_vectors)
        sols = surface.collide(lines)

        # Detect solutions outside the surface domain
        valid = torch.logical_and(sols <= surface.domain()[1], sols >= surface.domain()[0])
        
        # Filter data to keep only colliding rays
        sols = sols[valid]
        rays_origins = rays_origins[valid]
        rays_vectors = rays_vectors[valid]
        blocked = ~valid

        # TODO
        if valid is not None:
            input_masked = inputs.rays.masked(valid)
        else:
            input_masked = inputs.rays

        return OpticalData(input_masked, inputs.target, blocked, inputs.loss)


class OpticalSurface(Module):
    """
    Common base class for ReflectiveSurface and RefractiveSurface
    """

    def __init__(self, shape, scale=1., anchors=("origin", "origin")):
        super().__init__()

        self.shape = shape
        self.scale = scale
        self.anchors = anchors

    def surface(self, pos):
        return Surface(self.shape, pos=pos, scale=self.scale, anchor=self.anchors[0])

    def forward(self, inputs: OpticalData, sampling: dict):
        surface = self.surface(inputs.target)
        valid = None

        # special case for zero rays, TODO remove this and make sure the inner code works with N=0
        if inputs.rays.data.numel() == 0:
            collision_points = torch.empty((0, 0))
            output_rays = torch.empty((0, 0))
            blocked = None
        else:
            # For all rays, find the intersection with the surface and the normal vector at the intersection
            rays_origins, rays_vectors = (
                inputs.rays.get(["RX", "RY"]),
                inputs.rays.get(["VX", "VY"]),
            )
            lines = rays_to_coefficients(rays_origins, rays_vectors)
            sols = surface.collide(lines)

            # Detect solutions outside the surface domain
            valid = torch.logical_and(sols <= surface.domain()[1], sols >= surface.domain()[0])
            if False and torch.sum(~valid) > 0:
                raise RuntimeError("Some rays do not collide with the surface")

            # Filter data to keep only colliding rays
            sols = sols[valid]
            rays_origins = rays_origins[valid]
            rays_vectors = rays_vectors[valid]
            blocked = ~valid

            # Evaluate collision points and normals
            collision_points, surface_normals = surface.evaluate(sols), surface.normal(sols)

            # Verify no weirdness in the data
            assert torch.all(torch.isfinite(collision_points))
            assert torch.all(torch.isfinite(surface_normals))

            # Make sure collisions are in front of rays
            if False:
                ts = position_on_ray(rays_origins, rays_vectors, collision_points)
                if torch.any(ts <= 0):
                    print("warning: some ts <=0")
                if not torch.all(ts > 0): # TODO regression term on ts < 0 (== lens surface collision)
                    print("!! Some ts <= 0")
                    raise RuntimeError("negative collisions")

            # A surface always has two opposite normals, so keep the one pointing against the ray
            # i.e. the normal such that dot(normal, ray) < 0
            dot = torch.sum(surface_normals * rays_vectors, dim=1)
            collision_normals = torch.where((dot > 0).unsqueeze(1).expand(-1, 2), -surface_normals, surface_normals)

            # Verify no weirdness again
            assert torch.all(torch.isfinite(collision_normals))

            # Refract or reflect rays based on the derived class implementation
            output_rays = self.optical_function(rays_vectors, collision_normals)

        new_target = surface.at(self.anchors[1])

        # TODO
        if valid is not None:
            input_masked = inputs.rays.masked(valid)
        else:
            input_masked = inputs.rays

        # TODO
        if collision_points.numel() > 0:
            new_rays = input_masked.update(
                RX=collision_points[:, 0],
                RY=collision_points[:, 1],
                VX=output_rays[:, 0],
                VY=output_rays[:, 1],
            )
        else:
            new_rays = input_masked

        return OpticalData(new_rays, new_target, blocked, inputs.loss)


class ReflectiveSurface(OpticalSurface):
    def __init__(self, shape, scale=1., anchors=("origin", "origin")):
        super().__init__(shape, scale, anchors)
        

    def optical_function(self, rays, normals):
        return reflection(rays, normals)


class RefractiveSurface(OpticalSurface):
    def __init__(self, shape, n, scale=1., anchors=("origin", "origin")):
        super().__init__(shape, scale, anchors)
        self.n1, self.n2 = n
        
    def optical_function(self, rays, normals):
        return refraction(rays, normals, self.n1, self.n2, critical_angle='clamp')
