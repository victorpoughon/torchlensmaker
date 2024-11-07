import torch
import torch.nn as nn

from torchlensmaker.interp1d import interp1d
from torchlensmaker.shapes.common import line_coefficients
from torchlensmaker.shapes import Line


class PiecewiseLine:
    """
    Piecewise line starting at an implicit 0,0
    """

    def init_new(self, lens_radius, init):

        # Default zero init
        if init is None:
            X = torch.linspace(0., lens_radius, steps=2)[1:]
            Y = torch.zeros(1)

        else:
            X, Y = map(torch.as_tensor, init)

            if X[0] == 0.:
                raise ValueError("First point of PiecewiseLine is implicit, got X[0] = 0")
        
        assert X.shape == Y.shape, (X.shape, Y.shape)

        self.X = X
        self.params = {
            "Y": nn.Parameter(Y)
        }
    
    def init_share(self, share, scale):
        self.params = {}
    
    def __init__(self, lens_radius: float, init=None, share=None, scale=1.0):
        super().__init__()
        self.lens_radius = lens_radius

        if share is not None:
            self.init_share(share, scale)
        else:
            self.init_new(lens_radius, init)
        
        self.lens_radius = lens_radius
        self._share = share
        self._scale = scale
    
    def parameters(self):
        return self.params
    
    def domain(self):
        return torch.tensor([-self.lens_radius, self.lens_radius])
    
    def evaluate(self, X):
        cX, cY = self.coefficients()
        Y = interp1d(cX, cY, X)

        return torch.stack([X, Y], dim=-1)
    
    def coefficients(self):
        if self._share is None:
            param_X = self.X
            param_Y = self.params["Y"]
        else:
            param_X = self._share.X
            param_Y = self._share.params["Y"]

        X = torch.concatenate((torch.flip(-param_X, dims=[0]), torch.zeros(1), param_X)).contiguous()
        Y = torch.concatenate((torch.flip(param_Y, dims=[0]), torch.zeros(1), param_Y))

        assert X.numel() == Y.numel()

        return X, Y

    def edge_of_point(self, Px):
        """
        Given a point X coordinate,
        (or a list of points X coordinates)
        return the index of the edge the point falls into
        TODO deprecate, replace by searchsorted()
        """

        X, _ = self.coefficients()
        Ax = X[:-1]
        Bx = X[1:]
        within = torch.logical_or(
            torch.logical_and(Ax <= Px, Px < Bx),
            torch.logical_and(Bx <= Px, Px < Ax)
        ).to(dtype=int) # int to enable using argmax which doesn't support Bool
        if within.sum() == 0:
            raise RuntimeError("point none")
            return None
        else:
            return torch.argmax(within)


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

    def intersect_one(self, line):
        X, Y = self.coefficients()
        XY = torch.column_stack((X, Y))
        edges_coefficients = line_coefficients(XY[:-1, :], XY[1:, :])
        L = [Line(self.lens_radius, coefficients).intersect_batch(line.expand(1, -1)) for coefficients in edges_coefficients]
        intersection_points = torch.vstack(L)

        # True iff the intersection point is inside its corresponding edge segment
        Px = intersection_points[:, 0]    
        collision_index = self.edge_of_point(Px)
        
        return intersection_points[collision_index]

    def intersect_batch(self, lines):
        
        # TODO proper batch
        intersection_points = torch.zeros((lines.shape[0], 2))
        for i, line in enumerate(lines):
            intersection_points[i, :] = self.intersect_one(line)
        
        return intersection_points

    def collide(self, lines):
        """
        Collide the surface profile with lines.
        Returns collision points and normal unit vectors
        (arbitrarily oriented in either of the two possible directions)

        Args:
            lines: tensor (N, 3) - lines coefficients [a, b, c]

        Returns:
            points (N, 2), normals (N, 2)
        """

        points = self.intersect_batch(lines)
        normals = self.normal(points[:, 0])
        return points, normals
