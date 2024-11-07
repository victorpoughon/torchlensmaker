import math

import torch
import torch.nn as nn

from torchlensmaker.shapes.common import intersect_newton


class CircularArc:
    """
    An arc of circle

    Parameters:
        lens_radius: radius of the lens
        arc_radius: radius of curvature of the surface profile

        The sign of the radius indicates the direction of the center of curvature.

        The internal parameter is curvature = 1/radius, to allow optimization to
        cross over zero and change sign.
    """

    def share(self, scale=1.0):
        return type(self)(self.lens_radius, init=None, share=self, scale=scale)

    def __init__(self, lens_radius, init=None, share=None, scale=1.0):

        if init is not None and share is None:
            arc_radius = torch.atleast_1d(torch.as_tensor(init))
            assert torch.abs(arc_radius) >= lens_radius
            self.params = {
                "K": nn.Parameter(1./arc_radius)
            }
        elif init is None and share is not None:
            assert isinstance(share, CircularArc)
            self.params = {}
        
        self.lens_radius = lens_radius
        self._share = share
        self._scale = scale

    def parameters(self):
        return self.params

    def coefficients(self):
        if self._share is None:
            K = self.params["K"]
        else:
            K = self._share.params["K"]
        
        # Special case to avoid div by zero
        if torch.abs(K) < 1e-8:
            return torch.sign(K) * 1e8
        else:
            return 1. / K * self._scale

    def domain(self):
        R = self.coefficients()
        a = math.acos(self.lens_radius / torch.abs(R))
        if R > 0:
            return a, math.pi - a
        else:
            return -math.pi + a, -a

    def evaluate(self, ts):
        ts = torch.as_tensor(ts)
        R = self.coefficients()
        X = torch.abs(R)*torch.cos(ts)
        Y = torch.abs(R)*torch.sin(ts) - R
        return torch.stack((X, Y), dim=-1)

    def derivative(self, ts):
        R = self.coefficients()
        return torch.stack([
            - torch.abs(R) * torch.sin(ts),
            torch.abs(R) * torch.cos(ts)
        ], dim=-1)

    def normal(self, ts):
        "Normal vectors at the given parametric locations"

        deriv = self.derivative(ts)
        normal = torch.stack((-deriv[:, 1], deriv[:, 0]), dim=-1)
        return normal / torch.linalg.vector_norm(normal, dim=1).view((-1, 1))

    def newton_init(self, size):
        return torch.full(size, math.pi/2)
    
    def collide(self, lines):
        tn = intersect_newton(self, lines)
        return self.evaluate(tn), self.normal(tn)
