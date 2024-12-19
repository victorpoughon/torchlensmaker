import torch
import torch.nn as nn

from torchlensmaker.shapes.common import intersect_newton
from torchlensmaker.shapes import BaseShape


class Parabola(BaseShape):
    """
    Parabola of the form x = ay^2
    """

    def __init__(self, height, a):
        super().__init__()
        self.height = torch.as_tensor(height)
        self._a = torch.as_tensor(a)

        assert self._a.ndim == 0

    def coefficients(self):
        return self._a

    def parameters(self):
        if isinstance(self._a, nn.Parameter):
            return {"a": self._a}
        else:
            return {}

    def evaluate(self, y):
        y = torch.atleast_1d(torch.as_tensor(y))
        a = self.coefficients()
        x = a*torch.pow(y, 2)
        return torch.stack([x, y], dim=-1)

    def derivative(self, ys):
        return torch.stack((2 * self.coefficients() * ys, torch.ones_like(ys)), dim=1)

    def domain(self):
        r = self.height / 2
        return torch.tensor([-r, r])

    def normal(self, ys):
        # Compute the normal vectors, and normalize them
        normals = torch.stack(
            [torch.ones_like(ys), - 2 * self.coefficients() * ys], dim=1
        )
        return normals / torch.norm(normals, dim=1, keepdim=True)

    def newton_init(self, size):
        return torch.full(size, 0.)

    def collide(self, lines):
        # We could solve the equations directly,
        # but there is poor numerical stability when A ~ 0 and b ~ 0
        return intersect_newton(self, lines)
