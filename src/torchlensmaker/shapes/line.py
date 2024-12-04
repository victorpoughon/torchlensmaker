import torch
from torchlensmaker.shapes import BaseShape


class Line(BaseShape):
    """
    A straight line perpendicular to the principal axis
    """

    def __init__(self, width):
        self.width = width
    
    def parameters(self):
        return {}
    
    def coefficients(self):
        return None
    
    def domain(self):
        return torch.tensor([-self.width / 2, self.width / 2])

    def evaluate(self, X):
        X = torch.atleast_1d(torch.as_tensor(X))
        Y = torch.zeros_like(X)
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

        # Assume no horizontal lines, i.e. a != 0
        a, b, c = lines[:, 0], lines[:, 1], lines[:, 2]

        X = - c / a
        Y = torch.zeros_like(X)

        return torch.stack([X, Y], dim=-1)

    
    def collide(self, lines):
        points = self.intersect_batch(lines)
        normals = self.normal(points)
        return points, normals
