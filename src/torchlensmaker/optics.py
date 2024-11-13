import torch
import torch.nn as nn

from enum import Enum

from torchlensmaker.raytracing import (
    super_refraction,
    ray_point_squared_distance,
    position_on_ray,
    rays_to_coefficients
)


def loss_nonpositive(parameters, scale=1):
    return torch.where(parameters > 0, torch.pow(scale*parameters, 2), torch.zeros_like(parameters))


class FocalPointLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, rays, hook=None):
        ray_origins, ray_vectors = rays
        num_rays = ray_origins.shape[0]
        sum_squared = ray_point_squared_distance(ray_origins, ray_vectors, torch.zeros((2))).sum()
        return sum_squared / num_rays


class ParallelBeamUniform(nn.Module):
    def __init__(self, radius):
        super().__init__()
        self.radius = radius

    def forward(self, num_rays, hook=None):
        margin = 0.1
        rays_x = torch.linspace(-self.radius + margin, self.radius - margin, num_rays)
        rays_y = torch.zeros(num_rays)
        rays_origins = torch.column_stack((rays_x, rays_y))
        
        rays_vectors = torch.tile(torch.tensor([0., 1.]), (num_rays, 1))

        return (rays_origins, rays_vectors)


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


class FixedGap(nn.Module):
    "A fixed gap between two optical elements"

    def __init__(self, origin):
        super().__init__()
        self.origin = origin
    
    def forward(self, rays, hook=None):
        rays_origins, rays_vectors = rays
        return (
            rays_origins - torch.tensor([0.0, self.origin]),
            rays_vectors
        )


Anchor = Enum('Anchor', ['Center', 'Edge'])

class RefractiveSurface(nn.Module):
    def __init__(self, surface, n, anchors=(Anchor.Center, Anchor.Center)):
        super().__init__()

        self.surface = surface
        self.n1, self.n2 = n
        self.anchors = anchors

        # Register surface parameters
        for name, param in surface.parameters().items():
            if isinstance(param, nn.Parameter):
                self.register_parameter(name, param)

    
    def forward(self, rays, hook=None):
        rays_origins, rays_vectors = rays
        num_rays = rays_origins.shape[0]

        # Translate rays to surface origin using the front anchor
        anchor_edge_offset = self.surface.evaluate(self.surface.domain()[1:])[0][1]
        if self.anchors[0] == Anchor.Edge:
            rays_origins = rays_origins + torch.tensor([0.0, anchor_edge_offset])

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
        
        # Translate rays to the back anchor
        if self.anchors[1] == Anchor.Edge:
            collision_points = collision_points - torch.tensor([0.0, anchor_edge_offset])

        return (collision_points, collision_all_refracted)



class OpticalStack(nn.Module):
    def __init__(self, stack):
        super().__init__()
        self.stack = stack
        for i, module in enumerate(self.stack):
            self.add_module(str(i), module)
    
    def modules(self):
        return self.stack

    def forward(self, rays, hook=None):
        inputs = rays
        
        for optical_element in self.modules():
            output = optical_element.forward(inputs, hook)
            if hook:
                override = hook(optical_element, inputs, output)
                if override is not None:
                    output = override
            inputs = output

        return output


class Lens(OpticalStack):
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

    
