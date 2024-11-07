import math
import torch
import torch.nn as nn

from .raytracing import normed
import numpy as np

from .interp1d import interp1d


class Parabola:
    """
    Parabola of the form y = ax^2
    """

    def share(self, scale=1.0):
        return type(self)(self.lens_radius, init=None, share=self, scale=scale)

    def __init__(self, lens_radius, init=None, share=None, scale=1.0):
        super().__init__()
        self.lens_radius = lens_radius

        if init is not None and share is None:
            init = torch.atleast_1d(torch. as_tensor(init))
            param = nn.Parameter(init)
            self.params = {"a": param}
        elif init is None and share is not None:
            assert isinstance(share, Parabola)
            self.params = {}

        self.scale = scale
        self._share = share
    
    def coefficients(self):
        # Scaled coefficients for parameter sharing
        if self._share is None:
            return self.params["a"]
        else:
            return self._share.params["a"] * self.scale
    
    def parameters(self):
        return self.params

    def evaluate(self, x):
        x = torch.atleast_1d(torch.as_tensor(x))
        a = self.coefficients()
        y = a*torch.pow(x, 2)
        return torch.stack([x, y], dim=-1)

    def domain(self):
        "Return the stard and end points"

        r = self.lens_radius
        return torch.tensor([-r, r])

    def normal(self, point):
        return normed(torch.tensor([-2*self.coefficients()[0]*point[0], 1.]))

    def normal_batch(self, points):
        # Extract x coordinates
        x = points[:, 0]
        
        # Compute the normal vectors
        normals = torch.stack([-2*self.coefficients()[0]*x, 
                               torch.ones_like(x)], dim=1)
        
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
        normals = self.normal_batch(points)
        return points, normals


def line_coefficients(A, B):
    """
    Compute the coefficients (a, b, c) of the line equation ax + by + c = 0
    passing through points A and B for multiple pairs of points.
    
    Args:
    A: torch.Tensor of shape (N, 2) representing N points (x1, y1)
    B: torch.Tensor of shape (N, 2) representing N points (x2, y2)
    
    Returns:
    torch.Tensor of shape (N, 3) representing N sets of [a, b, c]
    """
    # Ensure inputs are tensors
    A = torch.as_tensor(A)
    B = torch.as_tensor(B)
    
    # Extract x and y coordinates
    x1, y1 = A[:, 0], A[:, 1]
    x2, y2 = B[:, 0], B[:, 1]
    
    # Compute a, b, c
    a = y2 - y1
    b = x1 - x2
    c = x2*y1 - x1*y2
    
    return torch.stack([a, b, c], dim=1)


class Line:

    def __init__(self, lens_radius, init=None, share=None, scale=1.0, optimize=True):
        if init is not None and share is None:
            init = torch.as_tensor(init)
            assert init.shape == (3,)
            if optimize:
                self.params = {
                    "abc": nn.Parameter(init)
                }
            else:
                self.params = {
                    "abc": init
                }
        else:
            raise NotImplementedError("")

        self.lens_radius = lens_radius
        self._share = share
        self._scale = scale
    
    def parameters(self):
        return self.params
    
    def coefficients(self):
        if self._share is None:
            return self.params["abc"]
        else:
            raise NotImplementedError("")
    
    def domain(self):
        return torch.tensor([-self.lens_radius, self.lens_radius])

    def evaluate(self, X):
        X = torch.atleast_1d(torch.as_tensor(X))
        # TODO doesn't work for vertical lines
        # TODO switch to parametric representation
        a, b, c = self.coefficients()
        Y = -a / b * X - c / b
        return torch.stack([X, Y], dim=-1)

    def normal(self, points):
        coefficients = self.coefficients()
        return torch.tile(normed(coefficients[:2]), (points.shape[0], 1))

    def intersect_batch(self, lines):
        """
        Intersect with multiple lines where lines is a tensor of shape (N, 3) 
        representing N lines in the form [a, b, c] coefficients.
        """
        # Ensure lines is a tensor
        lines = torch.as_tensor(lines)
        
        # Extract coefficients
        a1, b1, c1 = self.coefficients()
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
    
    def collide(self, lines):
        points = self.intersect_batch(lines)
        normals = self.normal(points)
        return points, normals


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


def mirror_points(A, B):
    "Mirror points A around points B"
    return torch.column_stack([
        2*B[:, 0] - A[:, 0],
        2*B[:, 1] - A[:, 1]
    ])


class BezierSpline:
    """
    Cubic Bezier Spline with constraints to represent a lens surface:
        - Symmetry around the Y axis
        - First knot is (0,0)
        - C1 continuity (control points are mirrored around knots)
        - First control point is on the y=0 line, so the tangent at (0,0) is horizontal
        - Knots' X coordinates are uniformly spread on the interval (0, radius)
        
    All methods expect the first dimension to be a batch dimension
    """

    # Constants for bezier curve matrix form
    M4 = torch.tensor([[1, 0, 0, 0], [-3, 3, 0, 0], [3, -6, 3, 0], [-1, 3, -3, 1]], dtype=torch.float32)
    M3 = torch.tensor([[1, 0, 0], [-2, 2, 0], [1, -2, 1]], dtype=torch.float32)


    @classmethod
    def from_parabola(cls, parabola):
        if not isinstance(parabola, Parabola):
            raise RuntimeError(f"expected type Parabola, got {type(parabola)}")
    
        # P1: 0, 0
        # P2: w/2, 0
        # P3: w, aw^2
        return 
    
    @classmethod
    def share(cls, other, scale=1.0):
        if not isinstance(other, BezierSpline):
            raise RuntimeError(f"expected type BezierSpline, got {type(other)}")
        
        return cls(other.radius, init=None, share=other, scale=scale)
    
    def parameters(self):
        return self.params
        
    def init_share(self, share, scale):
        self.params = {}
        self.num_intervals = share.num_intervals

    def init_new(self, width, init):

        # Default zero init
        if init is None:
            Y = torch.tensor([0.0])
            CX = torch.tensor([0.3*width, 1.3*width])
            CY = torch.tensor([0.0])
        else:
            Y, CX, CY = map(torch.as_tensor, init)
        
        assert Y.shape[0] + 1 == CX.shape[0] == CY.shape[0] + 1

        self.radius = width
        self.num_intervals = Y.shape[0]

        self.params = {
            "Y": nn.Parameter(Y),
            "CX": nn.Parameter(CX),
            "CY": nn.Parameter(CY),
        }

    def __init__(self, width, init=None, share=None, scale=1.0):
        """
        Create a new surface, either from 
        """
        
        if share is not None:
            self.init_share(share, scale)
        else:
            self.init_new(width, init)
        self.radius = width
        self.share = share
        self.scale = scale
    
    def coefficients(self):
        # Knots: X fixed on linspace, first Y fixed at zero
        if self.share is None:
            param_Y = self.params["Y"]
            param_CX = self.params["CX"]
            param_CY = self.params["CY"]
        else:
            param_Y = self.share.params["Y"] * self.scale
            param_CX = self.share.params["CX"]
            param_CY = self.share.params["CY"] * self.scale

        X = torch.linspace(0.0, self.radius, self.num_intervals + 1)
        Y = torch.cat((torch.zeros(1), param_Y))

        # Control points: X free, first Y fixed at zero
        CX = param_CX
        CY = torch.cat((torch.zeros(1), param_CY))

        assert X.numel() == self.num_intervals + 1
        assert Y.numel() == self.num_intervals + 1
        assert CX.numel() == self.num_intervals + 1
        assert CY.numel() == self.num_intervals + 1

        return X, Y, CX, CY

    def get_bezier_points(self, intervals):
        """
        Get the 4 bezier points for each interval index

        Args:
            intervals :: (N,)
        Returns:
            :: (N, 4, 2)
        """

        assert intervals.dim() == 1
        assert torch.all(intervals < self.num_intervals), intervals

        i = intervals
        X, Y, CX, CY = self.coefficients()
        this_knot = torch.stack([X[i], Y[i]], dim=-1)
        this_ctrl_point = torch.stack([CX[i], CY[i]], dim=-1)
        next_control_points = torch.column_stack([CX[i+1], CY[i+1]])
        next_knot = torch.column_stack([X[i+1], Y[i+1]])
        third_ctrl_point = mirror_points(next_control_points, next_knot)

        return torch.stack([
            this_knot,
            this_ctrl_point,
            third_ctrl_point,
            next_knot,
        ], dim=1)

    def domain(self):
        return torch.tensor([-self.num_intervals, self.num_intervals], dtype=torch.float32)
    
    def bezier_curve(self, ts):
        """
        Get the bezier curve four control points and t value for all spline positions ts
        ts are in ]M; M[ where M is the number of intervals

        Args:
            ts :: (N,)

        Returns: (points, t)
            points :: (N, 4, 2) The 4 control points
            t :: (N,) The t value in [0; 1]
        """

        M = self.num_intervals
        assert isinstance(ts, torch.Tensor) and ts.dim() == 1, ts
        assert torch.all(ts >= -M) and torch.all(ts <= M), ts

        # Compute the interval of each point
        intervals = torch.trunc(torch.abs(ts)).to(dtype=int)
        inner_ts = torch.frac(torch.abs(ts))

        # Special case for exact end points of the spline
        intervals = torch.where(torch.abs(ts) == M, M-1, intervals)
        inner_ts = torch.where(torch.abs(ts) == M, 1.0, inner_ts)

        points = self.get_bezier_points(intervals)

        # Symmetry around the Y axis
        # Where ts are negative, swap control points and t becomes 1-t
        points_swapped = points[:, [3, 2, 1, 0], :]
        points_mirrored = torch.stack((-points_swapped[:, :, 0], points_swapped[:, :, 1]), dim=-1)
        points = torch.where((ts < 0).view((-1, 1, 1)), points_mirrored, points)
        inner_ts = torch.where(ts < 0, 1.0 - inner_ts, inner_ts)

        return points, inner_ts
    

    def dump(self):
        X, Y, CX, CY = self.coefficients()
        print("X:", X)
        print("Y:", Y)
        print("CX:", CX)
        print("CY:", CY)
    
    def evaluate(self, ts):
        ""

        # Get the interval and fractional t position
        #i, t = self.get_interval(ts)
        #P = self.get_bezier_points(i)

        assert isinstance(ts, torch.Tensor) and ts.dim() == 1
        P, t = self.bezier_curve(ts)

        #  Evaluate the position using bezier curve matrix form  
        # T :: (N, 4)
        # M :: (4, 4)
        T = torch.stack([torch.full_like(t, 1.), t, torch.pow(t, 2), torch.pow(t, 3)], dim=-1)
        M = self.M4

        # points :: (N, 2)
        # points = T x M x P
        points = torch.bmm(T.unsqueeze(1) @ M, P).squeeze(1)
        return points

    def resample(self):
        X, Y, CX, CY = self.coefficients()

        z = 0.5
        h = z - 1.0
        Q1 = torch.tensor([
            [1, 0, 0, 0],
            [-h, z, 0, 0],
            [h**22, -2*h*z, z**2, 0],
            [-h**3, 3*h**2*z, -3*h*z**2, z**3],
        ], dtype=X.dtype)

        Q2 = torch.tensor([
            [-h**3, 3*h**2*z, -3*h*z**2, z**3],
            [0, h**2, -2*h*z, z**2],
            [0, 0, -h, z],
            [0, 0, 0, 1],
        ], dtype=X.dtype)

        M = self.num_intervals
        P, _ = self.bezier_curve(torch.arange(0, M) + 0.5)

        new_X = torch.zeros(2*M+1)
        new_Y = torch.zeros(2*M+1)
        new_CX = torch.zeros(2*M+1)
        new_CY = torch.zeros(2*M+1)

        for i in range(self.num_intervals):
            
            # New points of the first new half interval
            Pa = Q1 @ P[i, :, :]

            # New points of the second new half interval
            Pb = Q2 @ P[i, :, :]
            
            new_X[2*i] = Pa[0, 0]
            new_Y[2*i] = Pa[0, 1]
            new_CX[2*i] = Pa[1, 0]
            new_CY[2*i] = Pa[1, 1]

            new_X[2*i+1] = Pa[3, 0]
            new_Y[2*i+1] = Pa[3, 1]
            new_CX[2*i+1] = Pb[1, 0]
            new_CY[2*i+1] = Pb[1, 1]

            next_interval_ctrl_point = mirror_points(Pb[2].unsqueeze(0), Pb[3].unsqueeze(0))[0]
            
            new_X[2*i+2] = Pb[3, 0]
            new_Y[2*i+2] = Pb[3, 1]
            new_CX[2*i+2] = next_interval_ctrl_point[0]
            new_CY[2*i+2] = next_interval_ctrl_point[1]

        return BezierSpline(self.radius, init=(new_Y[1:], new_CX, new_CY[1:]))

    def wiggle(self, cx, cy, y):
        _, Y, CX, CY = self.coefficients()
        
        new_Y = Y + y*torch.randn_like(Y)
        new_CX = CX + cx*torch.randn_like(CX)
        new_CY = CY + cy*torch.randn_like(CY)

        return BezierSpline(self.radius, init=(new_Y[1:], new_CX, new_CY[1:]))

    def derivative(self, ts):
        "Evaluate the derivative at given parametric locations"
        
        # Get the control points and t value
        P, t = self.bezier_curve(ts)

        dA = 3*(P[:, 1] - P[:, 0])
        dB = 3*(P[:, 2] - P[:, 1])
        dC = 3*(P[:, 3] - P[:, 2])

        #  Evaluate the derivative using bezier curve matrix form  
        # T :: (N, 3)
        # M :: (3, 3)
        T = torch.stack([torch.full_like(t, 1.), t, torch.pow(t, 2)], dim=-1)
        M = self.M3
        P = torch.stack((dA, dB, dC), dim=1)

        # points :: (N, 2)
        # points = T x M x P
        points = torch.bmm(T.unsqueeze(1) @ M, P).squeeze(1)
        return points

    def normal(self, ts):
        "Normal vectors at the given parametric locations"

        deriv = self.derivative(ts)
        normal = torch.stack((-deriv[:, 1], deriv[:, 0]), dim=-1)
        return normal / torch.linalg.vector_norm(normal, dim=1).view((-1, 1))

    def newton_init(self, size):
        return torch.zeros(size)
    
    def collide(self, lines):
        tn = intersect_newton(self, lines)
        return self.evaluate(tn), self.normal(tn)


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


def newton_iteration(surface, lines, tn):
    # Compute value and derivative
    a, b, c = lines[:, 0], lines[:, 1], lines[:, 2]
    points = surface.evaluate(tn)
    diff = surface.derivative(tn)

    # Compute L and L'
    L = a*points[:, 0] + b*points[:, 1] + c
    Lp = a*diff[:, 0] + b*diff[:, 1]

    # Update solution
    delta = - L / Lp
    return tn + delta


def intersect_newton(surface, lines):
    """
    Intersect shape with N lines, using Newton's method

    Args:
        surface: the shape
        lines :: (N, 3): coefficients (a, b, c) of lines ax+by+c = 0

    Returns:
        ts :: (N,) parametric coordinate of each intersection
    """
    
    assert isinstance(lines, torch.Tensor) and lines.dim() == 2
   
    # Initialize solutions
    tn = surface.newton_init((lines.shape[0],))

    with torch.no_grad():
        for _ in range(10): # TODO parameters for newton iterations
            tn = newton_iteration(surface, lines, tn)

            # Clamp to the domain
            tn = torch.clamp(tn, *surface.domain())

    # One Newton iteration for backwards
    tn = newton_iteration(surface, lines, tn)
    # TODO, use this clamp to check for no collision?
    tn = torch.clamp(tn, *surface.domain())
    
    return tn
      