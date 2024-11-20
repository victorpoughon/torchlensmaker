import torch
import torch.nn as nn

from enum import Enum

from torchlensmaker.raytracing import (
    super_refraction,
    ray_point_squared_distance,
    position_on_ray,
    rays_to_coefficients
)

from torchlensmaker.surface import Surface


def loss_nonpositive(parameters, scale=1):
    return torch.where(parameters > 0, torch.pow(scale*parameters, 2), torch.zeros_like(parameters))


class FocalPointLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.pos = None

    def forward(self, inputs):

        (ray_origins, ray_vectors), target = inputs
        self.pos = target # store for rendering
        num_rays = ray_origins.shape[0]
        sum_squared = ray_point_squared_distance(ray_origins, ray_vectors, target).sum()
        return sum_squared / num_rays


class ParallelBeamUniform(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.width = width

    def forward(self, inputs):
        num_rays, target = inputs

        margin = 0.1 # TODO
        rays_x = torch.linspace(-self.width + margin, self.width - margin, num_rays)
        rays_y = torch.zeros(num_rays)
        
        rays_origins = target + torch.column_stack((rays_x , rays_y ))
        rays_vectors = torch.tile(torch.tensor([0., 1.]), (num_rays, 1))

        return ((rays_origins, rays_vectors), target)


class FixedGap(nn.Module):
    def __init__(self, offset):
        super().__init__()
        self.offset = offset
    
    def forward(self, inputs):
        rays, target = inputs
        return (rays, target + self.offset)


class ParallelBeamRandom(nn.Module):
    def __init__(self, radius):
        super().__init__()
        self.radius = radius

    def forward(self, num_rays, hook=None):
        rays_x = -self.radius + 2*self.radius * torch.rand(size=(num_rays,))
        rays_y = torch.zeros(num_rays,)
        rays_origins = torch.column_stack((rays_x, rays_y))

        rays_vectors = torch.tile(torch.tensor([0., 1.]), (num_rays, 1))

        return (rays_origins, rays_vectors)


class RefractiveSurface(nn.Module):
    def __init__(self, shape, n):
        super().__init__()

        self.shape = shape
        self.n1, self.n2 = n

    
    def forward(self, inputs):
        ((rays_origins, rays_vectors), target) = inputs
        num_rays = rays_origins.shape[0]

        self.surface = Surface(self.shape, pos=target)
         # could be not stored and rebuilt in the rendering code using target and shape

        collision_all_refracted = torch.zeros((num_rays, 2))

        # Get all rays intersections points with the surface and the normal vectors
        lines = rays_to_coefficients(rays_origins, rays_vectors)
        collision_points, surface_normals = self.surface.collide(lines)

        if not torch.all(torch.isfinite(collision_points)):
            print(self.surface.coefficients)
            print("lines", lines)
            print("colision points", collision_points)
            raise RuntimeError("nan detected in collision_points!")

        # Make sure collisions are in front of rays
        ts = position_on_ray(rays_origins, rays_vectors, collision_points)
        if torch.any(ts <= 0):
            print("warning: some ts <=0")
        if True and not torch.all(ts > 0): # TODO regression term on ts < 0 (== lens surface collision)
            print("!! Some ts <= 0")
            print("ts", ts)
            print("surface coeffs", self.surface.coefficients)
            print("rays_origins", rays_origins)
            print("rays_vectors", rays_vectors)
            print("collision_points", collision_points)
            print("lines", lines)
            raise RuntimeError("negative collisions")
        
        # A surface always has two opposite normals, so keep the one pointing against the ray
        # i.e. the normal such that dot(normal, ray) < 0
        dot = torch.sum(surface_normals * rays_vectors, dim=1)
        collision_normals = torch.where((dot > 0).unsqueeze(1).expand(-1, 2), -surface_normals, surface_normals)

        if torch.any(torch.isnan(collision_normals)):
            print("lines", lines)
            print("colision points", collision_points)
            print("normals", collision_normals)
            raise RuntimeError("nan detected in collision_normals!")
        
        # TODO batch refraction functions
        for index_ray in range(num_rays):
            # Refraction of rays
            #refracted_ray, fake_ray = clamped_refraction(rays_vectors[index_ray], collision_normal, self.n, 1.0), None
            try:
                refracted_ray, fake_ray = super_refraction(rays_vectors[index_ray], collision_normals[index_ray], self.n1, self.n2)
            except Exception as err:
                print("rays", rays_vectors[index_ray], collision_normals[index_ray])
                print("surface coeffs", self.surface.coefficients)
                print("n1 n2", self.n1, self.n2)
                raise err
                

            collision_all_refracted[index_ray, :] = refracted_ray

        return ((collision_points, collision_all_refracted), target)



class Lens:
    def __init__(self, surface1, gap, surface2):
        super().__init__([surface1, gap, surface2])
        self.surface1 = surface1
        self.gap = gap
        self.surface2 = surface2
    
    def thickness(self, r):
        "Return the lens total thickness at distance r from the center"
        pass
    
    def thickness(self):
        y1 = self.surface1.surface.evaluate(self.surface1.surface.domain())[1, 1]
        y2 = self.surface2.surface.evaluate(self.surface2.surface.domain())[1, 1]

        thickness_at_center = torch.as_tensor(self.gap.origin)
        if self.surface1.anchors[1] == Anchor.Edge:
            thickness_at_center += y1
        if self.surface2.anchors[0] == Anchor.Edge:
            thickness_at_center -= y2
        
        thickness_at_radius = torch.as_tensor(self.gap.origin)
        if self.surface1.anchors[1] == Anchor.Center:
            thickness_at_radius -= y1
        if self.surface2.anchors[0] == Anchor.Center:
            thickness_at_radius += y2

        thickness_at_radius = thickness_at_center - y1 + y2

        return thickness_at_center, thickness_at_radius

    
