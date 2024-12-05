import torch
import torch.nn as nn

from typing import Optional
from dataclasses import dataclass

from torchlensmaker.raytracing import (
    refraction,
    reflection,
    ray_point_squared_distance,
    position_on_ray,
    rays_to_coefficients
)

from torchlensmaker.surface import Surface


def loss_nonpositive(parameters, scale=1):
    return torch.where(parameters > 0, torch.pow(scale*parameters, 2), torch.zeros_like(parameters))

default_input = ((torch.tensor([]), torch.tensor([])), torch.zeros(2))


class FocalPointLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.pos = None

    def forward(self, inputs):

        (ray_origins, ray_vectors), target = inputs
        self.pos = target # store for rendering, TODO don't
        num_rays = ray_origins.shape[0]
        sum_squared = ray_point_squared_distance(ray_origins, ray_vectors, target).sum()
        return sum_squared / num_rays


class ParallelBeamUniform(nn.Module):
    def __init__(self, width, num_rays):
        super().__init__()
        self.width = width
        self.num_rays = num_rays

    def forward(self, inputs):
        (_, target) = inputs

        margin = 0.1 # TODO
        rays_x = torch.linspace(-self.width/2 + margin, self.width/2 - margin, self.num_rays)
        rays_y = torch.zeros(self.num_rays)
        
        rays_origins = target + torch.column_stack((rays_x , rays_y ))
        rays_vectors = torch.tile(torch.tensor([0., 1.]), (self.num_rays, 1))

        return ((rays_origins, rays_vectors), target)


class ParallelBeamRandom(nn.Module):
    def __init__(self, width, num_rays):
        super().__init__()
        self.width = width
        self.num_rays = num_rays

    def forward(self, inputs):
        (_, target) = inputs
    
        rays_x = -self.width / 2 + self.width * torch.rand(size=(self.num_rays,))
        rays_y = torch.zeros(self.num_rays,)
        rays_origins = torch.column_stack((rays_x, rays_y))

        rays_vectors = torch.tile(torch.tensor([0., 1.]), (self.num_rays, 1))

        return ((rays_origins, rays_vectors), target)


class Gap(nn.Module):
    def __init__(self, offset_y):
        super().__init__()
        self.offset_y = torch.as_tensor(offset_y)
    
    def forward(self, inputs):
        rays, target = inputs
        return (rays, target + torch.stack((torch.tensor(0.), self.offset_y)))


class OpticalSurface(nn.Module):
    """
    Common base class for ReflectiveSurface and RefractiveSurface
    """

    def __init__(self, shape, scale=1., anchors=("origin", "origin")):
        super().__init__()

        self.shape = shape
        self.scale = scale
        self.anchors = anchors

        # Technically, surface and pos are outputs of the computation done in forward(),
        # and should rather be returned and passed along the module stack.
        # But to preserve their location in the module tree and to enable nicer, semantic access to them
        # from rendering or optimization code, we store them in the module itself here.
        # That way, code can access them with syntax such as optics.lens1.surface1,
        # rather than output_surfaces[4].
        # But, they are only populated after an evaluation of the complete model.
        self.surface = None
        self.pos = None

    
    def forward(self, inputs):
        ((rays_origins, rays_vectors), target) = inputs
        num_rays = rays_origins.shape[0]

        self.pos = target
        self.surface = Surface(self.shape, pos=target, scale=self.scale, anchor=self.anchors[0])

        # For all rays, find the intersection with the surface and the normal vector at the intersection
        lines = rays_to_coefficients(rays_origins, rays_vectors)
        sols = self.surface.collide(lines)

        # Detect solutions outside the surface domain
        valid = torch.logical_and(sols <= self.surface.domain()[1], sols >= self.surface.domain()[0])
        if False and torch.sum(~valid) > 0:
            raise RuntimeError("Some rays do not collide with the surface")

        # Filter data to keep only colliding rays
        sols = sols[valid]
        rays_origins = rays_origins[valid]
        rays_vectors = rays_vectors[valid]

        # Evaluate collision points and normals
        collision_points, surface_normals = self.surface.evaluate(sols), self.surface.normal(sols)

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

        return ((collision_points, output_rays), self.surface.at(self.anchors[1]))


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
