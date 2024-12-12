import torch
import torch.nn as nn

from typing import Optional
from dataclasses import dataclass

from torchlensmaker.raytracing import (
    refraction,
    reflection,
    ray_point_squared_distance,
    position_on_ray,
    rays_to_coefficients,
    rot2d,
)

from torchlensmaker.surface import Surface
from torchlensmaker.shapes import Line


def loss_nonpositive(parameters, scale=1):
    return torch.where(parameters > 0, torch.pow(scale*parameters, 2), torch.zeros_like(parameters))


@dataclass
class OpticalData:
    """
    Holder class for the data that's passed between optical elements
    """

    # Tensor of shape (N, 2)
    # Rays origins points
    rays_origins: torch.Tensor

    # Tensor of shape (N, 2)
    # Rays unit vectors
    rays_vectors: torch.Tensor

    # Tensor of shape (2,)
    # Position of the next optical element
    target: torch.Tensor

    # None or tlm.Surface
    # Surface object of the previous optical element
    surface: Optional[Surface]

    # None or Tensor of shape (N,)
    # Mask array indicating which rays from the previous data in the optical
    # stack were blocked by the previous optical element
    blocked: Optional[torch.Tensor]


default_input = OpticalData(
    rays_origins = torch.empty((0, 2)),
    rays_vectors = torch.empty((0, 2)),
    target = torch.zeros(2),
    surface = None,
    blocked = None,
)

def focal_point_loss(data: OpticalData):
    num_rays = data.rays_origins.shape[0]
    sum_squared = ray_point_squared_distance(data.rays_origins, data.rays_vectors, data.target).sum()
    return sum_squared / num_rays


# TODO remove this or rename to FocalPoint
class FocalPointLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs: OpticalData):
        return inputs


class PointSource(nn.Module):
    def __init__(self, beam_angle, height=0):
        """
        height: height of the point source above the principal axis
        beam_angle: total angle of the emitted beam of rays (in degrees)
        """

        super().__init__()
        self.beam_angle = torch.deg2rad(
            torch.as_tensor(beam_angle, dtype=torch.float32)
        )
        self.height = torch.as_tensor(height, dtype=torch.float32)
        self.num_rays = 10

    def forward(self, inputs: OpticalData):
        # Create new rays by sampling the beam angle
        rays_origins = torch.tile(
            inputs.target + torch.tensor([0.0, self.height]), (self.num_rays, 1)
        )

        angles = torch.linspace(
            -self.beam_angle / 2, self.beam_angle / 2, self.num_rays
        )
        rays_vectors = rot2d(torch.tensor([1.0, 0.0]), angles)

        # Add new rays to the input rays
        return OpticalData(
            torch.cat((inputs.rays_origins, rays_origins), dim=0),
            torch.cat((inputs.rays_vectors, rays_vectors), dim=0),
            inputs.target,
            None,
            None,
        )


class PointSourceAtInfinity(nn.Module):
    def __init__(self, beam_diameter, angle=0):
        """
        beam_diameter: diameter of the beam of parallel light rays
        angle: angle of indidence with respect to the principal axis (in degrees)
        """

        super().__init__()
        self.beam_diameter = torch.as_tensor(beam_diameter, dtype=torch.float32)
        self.angle = torch.deg2rad(torch.as_tensor(angle, dtype=torch.float32))
        self.num_rays = 10

    def forward(self, inputs: OpticalData):
        # Create new rays by sampling the beam diameter
        margin = 0.1  # TODO
        rays_x = torch.zeros(self.num_rays)
        rays_y = torch.linspace(
            -self.beam_diameter / 2 + margin,
            self.beam_diameter / 2 - margin,
            self.num_rays,
        )

        rays_origins = inputs.target + torch.column_stack((rays_x, rays_y))
        vect = rot2d(torch.tensor([1.0, 0.0]), self.angle)
        rays_vectors = torch.tile(vect, (self.num_rays, 1))

        # Add new rays to the input rays
        return OpticalData(
            torch.cat((inputs.rays_origins, rays_origins), dim=0),
            torch.cat((inputs.rays_vectors, rays_vectors), dim=0),
            inputs.target,
            None,
            None,
        )


class Gap(nn.Module):
    def __init__(self, offset):
        super().__init__()
        self.offset = offset
    
    def forward(self, inputs: OpticalData):
        offset = torch.stack((torch.as_tensor(self.offset), torch.tensor(0.)))
        new_target = inputs.target + offset
        return OpticalData(inputs.rays_origins, inputs.rays_vectors, new_target, None, None)


class Aperture(nn.Module):
    def __init__(self, inner_width, outer_width):
        super().__init__()
        self.inner_width = inner_width
        self.outer_width = outer_width
        self.shape = Line(inner_width)
    
    def forward(self, inputs: OpticalData):
        surface = Surface(self.shape, pos=inputs.target)

        # TODO factor common collision code with OpticalSurface
        # For all rays, find the intersection with the surface and the normal vector at the intersection
        lines = rays_to_coefficients(inputs.rays_origins, inputs.rays_vectors)
        sols = surface.collide(lines)

        # Detect solutions outside the surface domain
        valid = torch.logical_and(sols <= surface.domain()[1], sols >= surface.domain()[0])
        
        # Filter data to keep only colliding rays
        sols = sols[valid]
        rays_origins = inputs.rays_origins[valid]
        rays_vectors = inputs.rays_vectors[valid]
        blocked = ~valid

        collision_points = surface.evaluate(sols)

        return OpticalData(collision_points, rays_vectors, inputs.target, surface, blocked)


class OpticalSurface(nn.Module):
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
    
    def forward(self, inputs: OpticalData):
        surface = self.surface(inputs.target)

        # special case for zero rays, TODO remove this and make sure the inner code works with B=0
        if inputs.rays_origins.numel() == 0:
            collision_points = torch.empty((0, 0))
            output_rays = torch.empty((0, 0))
            blocked = None
        else:
            # For all rays, find the intersection with the surface and the normal vector at the intersection
            lines = rays_to_coefficients(inputs.rays_origins, inputs.rays_vectors)
            sols = surface.collide(lines)

            # Detect solutions outside the surface domain
            valid = torch.logical_and(sols <= surface.domain()[1], sols >= surface.domain()[0])
            if False and torch.sum(~valid) > 0:
                raise RuntimeError("Some rays do not collide with the surface")

            # Filter data to keep only colliding rays
            sols = sols[valid]
            rays_origins = inputs.rays_origins[valid]
            rays_vectors = inputs.rays_vectors[valid]
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
        return OpticalData(collision_points, output_rays, new_target, surface, blocked)


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
