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
        self.pos = target # store for rendering, TODO don't
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
        rays_x = torch.linspace(-self.width/2 + margin, self.width/2 - margin, num_rays)
        rays_y = torch.zeros(num_rays)
        
        rays_origins = target + torch.column_stack((rays_x , rays_y ))
        rays_vectors = torch.tile(torch.tensor([0., 1.]), (num_rays, 1))

        return ((rays_origins, rays_vectors), target)


class ParallelBeamRandom(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.width = width

    def forward(self, inputs):
        num_rays, target = inputs
    
        rays_x = -self.width / 2 + self.width * torch.rand(size=(num_rays,))
        rays_y = torch.zeros(num_rays,)
        rays_origins = torch.column_stack((rays_x, rays_y))

        rays_vectors = torch.tile(torch.tensor([0., 1.]), (num_rays, 1))

        return ((rays_origins, rays_vectors), target)
    

class Gap(nn.Module):
    def __init__(self, offset):
        super().__init__()
        self.offset = torch.as_tensor(offset)
    
    def forward(self, inputs):
        rays, target = inputs
        return (rays, target + self.offset)


class GapY(nn.Module):
    def __init__(self, offset_y):
        super().__init__()
        self.offset_y = torch.as_tensor(offset_y)
    
    def forward(self, inputs):
        rays, target = inputs
        return (rays, target + torch.stack((torch.tensor(0.), self.offset_y)))


class GapX(nn.Module):
    def __init__(self, offset_x):
        super().__init__()
        self.offset_x = torch.as_tensor(offset_x)
    
    def forward(self, inputs):
        rays, target = inputs
        return (rays, target + torch.stack((self.offset_x, torch.tensor(0.))))


class RefractiveSurface(nn.Module):
    def __init__(self, shape, n, scale=1., anchors=("origin", "origin")):
        super().__init__()

        self.shape = shape
        self.n1, self.n2 = n
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

        collision_all_refracted = torch.zeros((num_rays, 2))

        # For all rays, find the intersection with the surface and the normal vector at the intersection
        lines = rays_to_coefficients(rays_origins, rays_vectors)
        collision_points, surface_normals = self.surface.collide(lines)

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
        
        # TODO batch refraction functions
        for index_ray in range(num_rays):
            # Refraction of rays
            #refracted_ray, fake_ray = clamped_refraction(rays_vectors[index_ray], collision_normal, self.n, 1.0), None
            try:
                refracted_ray = super_refraction(rays_vectors[index_ray], collision_normals[index_ray], self.n1, self.n2)
            except Exception as err:
                print("rays", rays_vectors[index_ray], collision_normals[index_ray])
                print("surface coeffs", self.surface.coefficients)
                print("n1 n2", self.n1, self.n2)
                raise err
                

            collision_all_refracted[index_ray, :] = refracted_ray

        return ((collision_points, collision_all_refracted), self.surface.at(self.anchors[1]))
