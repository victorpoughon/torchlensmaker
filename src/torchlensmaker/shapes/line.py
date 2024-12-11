import torch
from torchlensmaker.shapes import BaseShape


class Line(BaseShape):
    """
    A straight line segment perpendicular to the principal axis
    """

    def __init__(self, height):
        self.height = height
    
    def parameters(self):
        return {}
    
    def coefficients(self):
        return None
    
    def domain(self):
        return torch.tensor([-self.height / 2, self.height / 2])

    def evaluate(self, ts):
        Y = torch.atleast_1d(torch.as_tensor(ts))
        X = torch.zeros_like(Y)
        return torch.stack([X, Y], dim=-1)

    def normal(self, points):
        return torch.tile(torch.tensor([1., 0.]), (points.shape[0], 1))

    def intersect_batch(self, lines):
        """
        Intersect with multiple lines where lines is a tensor of shape (N, 3) 
        representing N lines in the form [a, b, c] coefficients.
        """
        # Ensure lines is a tensor
        lines = torch.as_tensor(lines)

        # Assume no vertical lines, i.e. b != 0
        a, b, c = lines[:, 0], lines[:, 1], lines[:, 2]

        return -c / b
    
    def collide(self, lines):
        return self.intersect_batch(lines)
