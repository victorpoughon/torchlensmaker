import torch
import torch.nn as nn

from torchlensmaker.shapes.common import mirror_points, intersect_newton

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

