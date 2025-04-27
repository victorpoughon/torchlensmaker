import math
import torch
import torch.nn as nn
import torchlensmaker as tlm
import numpy as np

from torchlensmaker.core.collision_detection import LM

from typing import Any


import matplotlib.pyplot as plt



def find_closest_points(surface, P, V):
    N = P.shape[0]
    
    t = torch.zeros(N).unsqueeze(0)
    algo = LM(0.5)

    with torch.no_grad():
        for i in range(30):
            t = t - algo.delta(surface, P, V, t, max_delta=100.0)

    t = t - algo.delta(surface, P, V, t, max_delta=100.0)

    return t.squeeze(0)

def loss(surface, P, V):
    t = find_closest_points(surface, P, V)
    points = P + t.unsqueeze(-1).expand_as(V) * V
    F = surface.F(points)
    return F.sum() / points.shape[0]




class XXLightSource(nn.Module):
    def __init__(self, all_rays):
        super().__init__()
        self.all_rays = all_rays

    @classmethod
    def load(cls, half: bool = False):
        data = np.loadtxt('xx_lighttools.txt', usecols=(0, 1, 2, 3, 4, 5))
        all_rays = torch.from_numpy(data).float()

        if half:
            all_rays = all_rays[all_rays[:, 3] > 0.]
            assert all_rays.shape == (500234, 6), all_rays.shape
        else:
            assert all_rays.shape == (1000000, 6)
        
        return cls(all_rays)

    def forward(self, inputs):

        # Pick random rays along the 'xx' dimension
        N = inputs.sampling['xx'].N
        indices = torch.randint(0, self.all_rays.shape[0], (N,))
        rays = self.all_rays[indices]

        P = 1000*torch.stack((torch.zeros_like(rays[:, 0]), rays[:, 0], rays[:, 1]), -1)
        V = torch.stack((-rays[:, 5], rays[:, 3], rays[:, 4]), -1)

        V = torch.nn.functional.normalize(V, dim=-1)
    
        # Apply kinematic transform
        P, V = inputs.tf().direct_rays(P, V)
        
        return inputs.replace(
            P=torch.cat((inputs.P, P), dim=0),
            V=torch.cat((inputs.V, V), dim=0),
        )


class NonImagingRod(nn.Module):
    def __init__(self, surface):
        self.surface = surface
        super().__init__()

    def forward(self, inputs: tlm.OpticalData) -> tlm.OpticalData:
        N, dim = inputs.P.shape
        assert dim == 3

        if N == 0:
            raise RuntimeError("No rays")

        # Convert rays to surface local frame
        P, V = inputs.P, inputs.V
        transform = tlm.forward_kinematic(inputs.transforms)
        P_local = transform.inverse_points(inputs.P)
        V_local = transform.inverse_vectors(inputs.V)
        
        t = find_closest_points(self.surface, P_local, V_local)

        # Closest points
        points_local = P_local + t.unsqueeze(-1).expand_as(V_local) * V_local
        #points_global = P + t.unsqueeze(-1).expand_as(V) * V

        loss = self.surface.F(points_local).sum()

        return inputs.replace(loss=inputs.loss + loss)


class RodArtist:
    def render_module(self, collective, module) -> list[Any]:
        tf = collective.input_tree[module].tf()
        s = module.surface
        return [{
            "type": "surface",
            "bcyl": [s.xmin.item(), s.xmax.item(), s.tau.item()],
            "matrix": tf.hom_matrix().tolist()
        }]

    def render_rays(self, collective, module) -> list[Any]:
        inputs = collective.input_tree[module]
        # render rays until closest point?
        return []


class RaysViewerPlane(nn.Module):
    "Utility component to vizualize rays distribution"

    def __init__(self, diameter, title):
        super().__init__()
        self.title = title
        surface = tlm.CircularPlane(diameter, dtype=torch.float64)
        self.collision_surface = tlm.CollisionSurface(surface)

    def forward(self, inputs):

        # Collision detection
        t, normals, valid_collision, new_chain = self.collision_surface(inputs)
        collision_points = inputs.P + t.unsqueeze(-1).expand_as(inputs.V) * inputs.V

        if "disable_viewer" not in inputs.sampling:
            with torch.inference_mode():
                # compute rays intersection with X=0 plane
                # plot YZ linear distribution
                # plot MN angular distribution

                X, Y, Z = collision_points.unbind(-1)
                L, M, N = inputs.V.unbind(-1)

                f, (ax1, ax2) = plt.subplots(1, 2)

                ax1.scatter(Y, Z, s=0.2)
                ax1.set_aspect("equal")
                ax2.scatter(M, N, s=0.2)
                ax2.set_aspect("equal")
                f.suptitle(self.title)
        
        return inputs
