import torch
import torch.nn as nn

from torchlensmaker.interp1d import interp1d
from torchlensmaker.shapes.common import line_coefficients
from torchlensmaker.shapes import BaseShape


def line_lines_intersection(coefficients, lines):
        """
        Intersect the line coefficients = a,b,c
        with multiple lines where lines is a tensor of shape (N, 3) 
        representing N lines in the form [a, b, c] coefficients.
        """
        # Ensure lines is a tensor
        lines = torch.as_tensor(lines)
        
        # Extract coefficients
        a1, b1, c1 = coefficients
        a2, b2, c2 = lines[:, 0], lines[:, 1], lines[:, 2]
        
        # Compute determinant
        det = a1 * b2 - a2 * b1
        
        # Prepare the result tensor
        result = torch.zeros((lines.shape[0], 2), dtype=torch.float32)
        
        # Compute intersection points where det != 0
        valid = torch.abs(det) >= 1e-8
        x = torch.zeros_like(det)
        y = torch.zeros_like(det)
        
        x[valid] = (b1 * c2[valid] - b2[valid] * c1) / det[valid]
        y[valid] = (c1 * a2[valid] - c2[valid] * a1) / det[valid]
        
        result[:, 0] = x
        result[:, 1] = y
        
        # Set intersection to [inf, inf] where lines are parallel
        result[~valid] = float('inf')
        
        return result


class PiecewiseLine(BaseShape):
    """
    Piecewise line starting at an implicit 0,0
    """

    
    def __init__(self, width: float, Y):
        """
        Y is height of each point
        first 0 is implicit
        """

        super().__init__()
        self.width = width
        self._Y = torch.as_tensor(Y)
    
    def coefficients(self):
        N = self._Y.shape[0]
        param_X = torch.linspace(0., self.width/2, steps=N+1)[1:]
        param_Y = self._Y

        X = torch.concatenate((torch.flip(-param_X, dims=[0]), torch.zeros(1), param_X)).contiguous()
        Y = torch.concatenate((torch.flip(param_Y, dims=[0]), torch.zeros(1), param_Y))

        assert X.numel() == Y.numel()

        return X, Y

    def parameters(self):
        if isinstance(self._Y, nn.Parameter):
            return {"Y": self._Y}
        else:
            return {}
    
    def domain(self):
        return torch.tensor([-self.width/2, self.width/2])
    
    def evaluate(self, X):
        cX, cY = self.coefficients()
        Y = interp1d(cX, cY, X)
        return torch.stack([X, Y], dim=-1)

    def interval_index(self, xs):
        """
        Given X coordinates of points: xs
        Return the index of the edge the point falls into
        """

        X, Y = self.coefficients()
        
        # find intervals
        indices = torch.searchsorted(X, xs.contiguous())

        # special case for newX == X[0]
        indices = torch.where(xs == X[0], 1, indices)

        # -1 here because we want the start of the interval
        return indices - 1
        
    def normal(self, xs):
        cX, cY = self.coefficients()
        XY = torch.column_stack((cX, cY))
        edges_coefficients = line_coefficients(XY[:-1, :], XY[1:, :])
        
        indices = self.interval_index(xs)
        normals = torch.nn.functional.normalize(edges_coefficients[:, :2], dim=1)

        return normals[indices]


    def intersect_batch(self, lines):
        X, Y = self.coefficients()
        XY = torch.column_stack((X, Y))
        edges_coefficients = line_coefficients(XY[:-1, :], XY[1:, :])
        
        # default value out of domain
        default = self.domain()[0] * 1.1
        collisions = torch.full((lines.shape[0],), default)

        # collide each segment with all rays
        for i, coefficients in enumerate(edges_coefficients):
            
            # collisions with this segment's full line
            col = line_lines_intersection(coefficients, lines)[:, 0]

            # store the collisions that fall within the current segement
            index = self.interval_index(col)
            mask = index == i
            collisions[mask] = col[mask]

        return collisions

    def collide(self, lines):
        return self.intersect_batch(lines)

