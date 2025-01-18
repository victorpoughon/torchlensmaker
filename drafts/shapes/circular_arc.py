import math

import torch
import torch.nn as nn

from torchlensmaker.shapes.common import intersect_newton
from torchlensmaker.shapes import BaseShape


class CircularArc(BaseShape):
    """
    An arc of circle parametrized by the Y coordinate

    X = (K * t**2) / (1 + sqrt(1 - t**2 * K**2))
    Y = t

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
        if torch.abs(K) < 1e-6:
            return torch.sign(K) * 1e6
        else:
            return 1. / K

    def domain(self):
        return -self.height / 2, self.height / 2

    def evaluate(self, ts):
        ts = torch.as_tensor(ts)

        K = self._K
        
        Y = ts
        X = (K * ts**2) / (1 + torch.sqrt(1 - ts**2 * K**2))

        return torch.stack((X, Y), dim=-1)

    def derivative(self, ts):
        K = self._K

        Yp = torch.ones_like(ts)
        Xp = K*ts / torch.sqrt( 1 - ts ** 2 * K **2)

        return torch.stack([Xp, Yp], dim=-1)

    def normal(self, ts):
        deriv = self.derivative(ts)
        normal = torch.stack((-deriv[:, 1], deriv[:, 0]), dim=-1)
        return normal / torch.linalg.vector_norm(normal, dim=1).view((-1, 1))

    def newton_init(self, size):
        return torch.full(size, 0.)

    def collide(self, lines):
        return intersect_newton(self, lines)
