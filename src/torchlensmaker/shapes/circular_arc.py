import math

import torch
import torch.nn as nn

from torchlensmaker.shapes.common import intersect_newton
from torchlensmaker.shapes import BaseShape


class CircularArc(BaseShape):
    """
    An arc of circle

    Parameters:
        height: total height of the shape accross the principal axis (typically diameter of the lens or mirror)
        arc_radius: radius of curvature of the shape

        The sign of the radius indicates the direction of the center of curvature:
        * radius > 0: the arc bends towards the positive X axis
        * radius < 0: the arc bends towards the negative X axis

        The internal parameter is curvature = 1/radius, to allow optimization to
        cross over zero and change sign.
    """

    def __init__(self, height, r):
        assert torch.abs(torch.as_tensor(r)) >= height / 2

        if isinstance(r, nn.Parameter):
            self._K = nn.Parameter(torch.tensor(1./r.item()))
        else:
            self._K = torch.as_tensor(1./r)

        self.height = height

    def parameters(self):
        if isinstance(self._K, nn.Parameter):
            return {"K": self._K}
        else:
            return {}

    def coefficients(self):
        K = self._K

        # Special case to avoid div by zero
        if torch.abs(K) < 1e-8:
            return torch.sign(K) * 1e8
        else:
            return 1. / K

    def domain(self):
        R = self.coefficients()
        a = math.asin(self.height / (2 * torch.abs(R)))
        return -a, +a


    def evaluate(self, ts):
        ts = torch.as_tensor(ts)

        R = self.coefficients()
        if R > 0:
            ts = ts + math.pi

        X = torch.abs(R)*torch.cos(ts) + R
        Y = torch.abs(R)*torch.sin(ts)
        return torch.stack((X, Y), dim=-1)

    def derivative(self, ts):
        R = self.coefficients()
        if R > 0:
            ts = ts + math.pi

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
        return torch.full(size, 0.)

    def collide(self, lines):
        return intersect_newton(self, lines)
