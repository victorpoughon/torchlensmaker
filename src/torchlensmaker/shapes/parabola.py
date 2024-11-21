import torch
import torch.nn as nn

from torchlensmaker.shapes.common import normed
from torchlensmaker.shapes import BaseShape


class Parabola(BaseShape):
    """
    Parabola of the form y = ax^2
    """

    def __init__(self, width, a):
        super().__init__()
        self.width = width
        self._a = torch.as_tensor(a)

        assert self._a.ndim == 0
    
    def coefficients(self):
        return self._a

    def parameters(self):
        if hasattr(self, "_a") and isinstance(self._a, nn.Parameter):
            return {"a": self._a}
        else:
            return {}

    def evaluate(self, x):
        x = torch.atleast_1d(torch.as_tensor(x))
        a = self.coefficients()
        y = a*torch.pow(x, 2)
        return torch.stack([x, y], dim=-1)

    def domain(self):
        "Return the start and end points"

        r = self.width / 2
        return torch.tensor([-r, r])

    def normal(self, xs):
        # Compute the normal vectors
        normals = torch.stack([-2*self.coefficients()*xs, 
                               torch.ones_like(xs)], dim=1)
        
        # Normalize the vectors
        normalized_normals = normals / torch.norm(normals, dim=1, keepdim=True)
        
        return normalized_normals
    
    def intersect_batch(self, lines):
        """
        Compute the intersection points with multiple lines
        Line are assumed to not be horizontal
        The solution returned is the one closest to x=0
        
        lines: Tensor (N, 3) - coefficient of the lines in the form ax + by + c = 0
        
        Returns:
            Tensor (N, 2) - intersection points for each line
        """
        
        # TODO change to work with all line orientations (parametric line?)

        # Ensure input is a tensor
        lines = torch.as_tensor(lines)  
        
        # Extract line coefficients
        a, b, c = lines[:, 0], lines[:, 1], lines[:, 2]
        A = self.coefficients()

        # This implementation is based on the 'citardauq' formula for quadratic polynomials
        # as it gives good numerical stability when both A (the parabola coefficient)
        # and b (the second line coefficient) are close to zero.
        # See https://en.wikipedia.org/wiki/Quadratic_formula#Square_root_in_the_denominator

        # Avoid sqrt(<0) and divide by zero
        # where() and grad is tricky, make sure to use the "double where()" trick
        # See https://github.com/tensorflow/probability/blob/main/discussion/where-nan.pdf
        delta = torch.pow(a, 2) - 4*b*A*c
        sqrt_delta = torch.sqrt(torch.where(delta >= 0, delta, 1.0))

        inner1 = -a - sqrt_delta
        inner2 = -a + sqrt_delta

        denom1 = torch.where(torch.isclose(inner1, torch.zeros(1)), 1.0, inner1)
        denom2 = torch.where(torch.isclose(inner2, torch.zeros(1)), 1.0, inner2)

        # The root we want depends on the sign of a
        x = torch.where(a >= 0,
            2*c / denom1,
            2*c / denom2,
        )

        y = A*torch.pow(x, 2)

        return torch.stack((x, y), dim=1)
    
    def collide(self, lines):
        points = self.intersect_batch(lines)
        normals = self.normal(points[:, 0])
        return points, normals
