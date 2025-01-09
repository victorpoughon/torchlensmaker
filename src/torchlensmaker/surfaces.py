import torchlensmaker as tlm
import torch

# shorter for type annotations
Tensor = torch.Tensor


class LocalSurface3D:
    """
    Defines a 3D surface in a local reference frame
    """

    def __init__(self, outline: tlm.Outline):
        self.outline = outline

    def local_collide(self, P: Tensor, V: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Find collision points and surface normals of ray-surface intersection
        for parametric rays P+tV expressed in the surface local frame.

        Returns:
            t: Value of parameter t such that P + tV is on the surface
            normals: Normal vectors to the surface at the collision points
            valid: Bool tensor indicating which rays do collide with the surface
        """
        raise NotImplementedError

    def extent(self) -> Tensor:
        """
        Extent along the X axis
        i.e. X coordinate of the point on the surface such that |X| is maximized
        """
        raise NotImplementedError

    def contains(self, points: Tensor, tol: float = 1e-6) -> Tensor:
        raise NotImplementedError


class Plane(LocalSurface3D):
    "X=0 plane"

    def __init__(self, outline: tlm.Outline):
        super().__init__(outline)

    def samples2D(self, N: int) -> Tensor:
        r = torch.linspace(0, self.outline.max_radius(), N)
        return torch.stack((torch.zeros(N), r), dim=-1)

    def local_collide(self, P: Tensor, V: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        t = -P[:, 0] / V[:, 0]
        local_points = P + t.unsqueeze(1).expand((-1, 3)) * V
        local_normals = torch.tile(Tensor([-1.0, 0.0, 0.0]), (P.shape[0], 1))
        valid = self.outline.contains(local_points)
        return t, local_normals, valid

    def extent(self) -> Tensor:
        return torch.zeros(1)

    def contains(self, points: Tensor, tol: float = 1e-6) -> Tensor:
        return torch.logical_and(
            self.outline.contains(points), torch.abs(points[:, 0]) < tol
        )


class SquarePlane(Plane):
    def __init__(self, side_length: float):
        super().__init__(tlm.SquareOutline(side_length))


class CircularPlane(Plane):
    "aka disk"

    def __init__(self, diameter: float):
        super().__init__(tlm.CircularOutline(diameter))


class ImplicitSurface3D(LocalSurface3D):
    """
    Surface3D defined in implicit form: F(x,y,z) = 0
    """

    def __init__(self, outline: tlm.Outline):
        super().__init__(outline)

    def contains(self, points: Tensor, tol: float = 1e-6) -> Tensor:
        return torch.logical_and(
            self.outline.contains(points), torch.abs(self.F(points)) < tol
        )

    def local_collide(self, P: Tensor, V: Tensor) -> tuple[Tensor, Tensor, Tensor]:

        # Initial guess is the intersection of rays with the X=0 plane
        init_t = -P[:, 0] / V[:, 0]

        t = intersect_newton_3D(self, P, V, init_t)

        local_points = P + t.unsqueeze(1).expand((-1, 3)) * V
        local_normals = self.F_grad(local_points)

        # If there is no intersection, newton's method won't converge
        # and points will not be on the surface
        # So verify intersection here and filter points
        # that aren't on the surface
        valid = self.contains(local_points, tol=1e-3)

        return t, local_normals, valid

    def F(self, points: Tensor) -> Tensor:
        """
        Implicit equation for the 3D shape: F(x,y,z) = 0

        Args:
            points: tensor of shape (N, 3) where columns are X, Y, Z coordinates and N is the batch dimension

        Returns:
            F: value of F at the given points, tensor of shape (N,)
        """
        raise NotImplementedError

    def F_grad(self, points: Tensor) -> Tensor:
        """
        Gradient of F

        Args:
            points: tensor of shape (N, 3) where columns are X, Y, Z coordinates and N is the batch dimension

        Returns:
            F_grad: value of the gradient of F at the given points, tensor of shape (N, 3)
        """
        raise NotImplementedError


class Parabola(ImplicitSurface3D):
    def __init__(self, diameter: float, a: float):
        super().__init__(tlm.CircularOutline(diameter))
        self.a = a

    def samples2D(self, N: int) -> Tensor:
        """
        Generate N sample points located on the shape's curve with r >= 0
        """

        r = torch.linspace(0, self.outline.max_radius(), N)
        x = self.a * r**2
        return torch.stack((x, r), dim=-1)

    def extent(self) -> Tensor:
        r = self.outline.max_radius()
        return torch.as_tensor(self.a * r**2)

    def f(self, x: Tensor, r: Tensor) -> Tensor:
        return torch.mul(self.a, torch.pow(r, 2) - x)

    def f_grad(self, x: Tensor, r: Tensor) -> Tensor:
        return torch.stack((-torch.ones_like(x), 2 * self.a * r), dim=-1)

    def F(self, points: Tensor) -> Tensor:
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        return torch.mul(self.a, (y**2 + z**2)) - x

    def F_grad(self, points: Tensor) -> Tensor:
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        return torch.stack(
            (-torch.ones_like(x), 2 * self.a * y, 2 * self.a * z), dim=-1
        )


class Sphere(ImplicitSurface3D):
    def __init__(self, diameter: float, r: float):
        super().__init__(tlm.CircularOutline(diameter))
        assert (
            torch.abs(torch.as_tensor(r)) >= diameter / 2
        ), f"Sphere diameter ({diameter}) must be less than 2x its arc radius (2x{r}={2*r})"
        self.diameter = diameter
        self.K = torch.as_tensor(1.0 / r)

    def extent(self) -> Tensor:
        r = self.outline.max_radius()
        K = self.K
        return torch.div(K * r, 1 + torch.sqrt(1 - r * K**2))

    def samples2D(self, N: int) -> Tensor:
        K = self.K
        r = torch.linspace(0, self.outline.max_radius(), N)
        x = (K * r**2) / (1 + torch.sqrt(1 - r**2 * K**2))
        return torch.stack((x, r), dim=-1)

    def F(self, points: Tensor) -> Tensor:
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        K = self.K
        r2 = y**2 + z**2
        return torch.div(K * r2, 1 + torch.sqrt(1 - r2 * K**2)) - x

    def F_grad(self, points: Tensor) -> Tensor:
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        K = self.K
        r2 = y**2 + z**2
        denom = torch.sqrt(1 - r2 * K**2)
        return torch.stack(
            (-torch.ones_like(x), (K * y) / denom, (K * z) / denom), dim=-1
        )


def newton_delta(surface: ImplicitSurface3D, P: Tensor, V: Tensor, t: Tensor) -> Tensor:
    "Compute the delta for one step of Newton's method"

    points = P + t.unsqueeze(1).expand_as(V) * V

    F = surface.F(points)
    F_grad = surface.F_grad(points)

    # Denominator will be zero if F_grad and V are orthogonal
    denom = torch.sum(F_grad * V, dim=1)

    return F / denom


def intersect_newton_3D(
    surface: ImplicitSurface3D, P: Tensor, V: Tensor, init_t: Tensor
) -> Tensor:
    """
    Collision detection of parametric rays with implicit surface using Newton's
    method.

    Rays are defined by P + tV where P are origin points, and V and unit length
    direction vectors.

    Args:
        P: tensor (N, 3), rays origin points
        V: tensor (N, 3), rays unit vectors
        init_t: tensor (N,), initial value for t

    Returns:
        t: tensor (N,), t values after Newton iterations
    """

    assert isinstance(P, Tensor) and P.dim() == 2
    assert isinstance(V, Tensor) and V.dim() == 2
    assert P.shape[0] == V.shape[0]
    assert P.shape[1] == V.shape[1] == 3

    # Initialize solutions t
    t = init_t

    with torch.no_grad():
        for _ in range(20):  # TODO parameters for newton iterations
            # TODO warning if stopping due to max iter (didn't converge)
            delta = newton_delta(surface, P, V, t)
            # TODO early stop if delta is small enough
            t = t - delta

    # One newton iteration for backwards pass
    t = t - newton_delta(surface, P, V, t)

    return t