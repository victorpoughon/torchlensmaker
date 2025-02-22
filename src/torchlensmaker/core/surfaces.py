import torch
import torch.nn as nn

from torchlensmaker.core.outline import (
    Outline,
    SquareOutline,
    CircularOutline,
)

from torchlensmaker.core.sphere_sampling import (
    sphere_samples_angular,
    sphere_samples_linear,
)
from torchlensmaker.core.tensor_manip import to_tensor

from torchlensmaker.core.collision_detection import CollisionAlgorithm, Newton

from typing import Iterable

# shorter for type annotations
Tensor = torch.Tensor


class LocalSurface:
    """
    Defines a surface in a local reference frame
    """

    def __init__(self, outline: Outline, dtype: torch.dtype = torch.float64):
        self.outline = outline
        self.dtype = dtype

    def parameters(self) -> dict[str, nn.Parameter]:
        raise NotImplementedError

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

    def extent_x(self) -> Tensor:
        """
        Extent along the X axis
        i.e. X coordinate of the point on the surface such that |X| is maximized
        """
        raise NotImplementedError

    def extent(self, dim: int, dtype: torch.dtype) -> Tensor:
        "N-dimensional extent point"
        return torch.cat(
            (self.extent_x().unsqueeze(0), torch.zeros(dim - 1, dtype=dtype)),
            dim=0,
        )

    def zero(self, dim: int, dtype: torch.dtype) -> Tensor:
        "N-dimensional zero point"
        return torch.zeros((dim,), dtype=dtype)

    def contains(self, points: Tensor, tol: float = 1e-6) -> Tensor:
        raise NotImplementedError

    def testname(self) -> str:
        "A string for identification in test cases"
        raise NotImplementedError


class Plane(LocalSurface):
    "X=0 plane"

    def __init__(self, outline: Outline, dtype: torch.dtype):
        super().__init__(outline, dtype)

    def parameters(self) -> dict[str, nn.Parameter]:
        return {}

    def samples2D(self, N: int) -> Tensor:
        r = torch.linspace(0, self.outline.max_radius(), N)
        return torch.stack((torch.zeros(N), r), dim=-1)

    def local_collide(self, P: Tensor, V: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        dim = P.shape[1]
        t = -P[:, 0] / V[:, 0]
        local_points = P + t.unsqueeze(1).expand_as(V) * V
        normal = (
            torch.tensor([-1.0, 0.0], dtype=self.dtype)
            if dim == 2
            else torch.tensor([-1.0, 0.0, 0.0], dtype=self.dtype)
        )
        local_normals = torch.tile(normal, (P.shape[0], 1))
        valid = self.outline.contains(local_points)
        return t, local_normals, valid

    def extent_x(self) -> Tensor:
        return torch.as_tensor(0.0, dtype=self.dtype)

    def contains(self, points: Tensor, tol: float = 1e-6) -> Tensor:
        return torch.logical_and(
            self.outline.contains(points), torch.abs(points[:, 0]) < tol
        )


class SquarePlane(Plane):
    def __init__(self, side_length: float, dtype: torch.dtype = torch.float64):
        super().__init__(SquareOutline(side_length), dtype)

    def testname(self) -> str:
        return f"SquarePlane-{self.side_length}"


class CircularPlane(Plane):
    "aka disk"

    def __init__(self, diameter: float, dtype: torch.dtype = torch.float64):
        super().__init__(CircularOutline(diameter), dtype)

    def testname(self) -> str:
        return f"CircularPlane-{self.diameter}"


class ImplicitSurface(LocalSurface):
    """
    Surface3D defined in implicit form: F(x,y,z) = 0
    """

    def __init__(self, collision: CollisionAlgorithm = Newton(15, 0.8), **kwargs):
        super().__init__(**kwargs)
        self.collision_algorithm = collision

    def contains(self, points: Tensor, tol: float = 1e-6) -> Tensor:
        dim = points.shape[1]

        F = self.F if dim == 3 else self.f

        return torch.logical_and(
            self.outline.contains(points), torch.abs(F(points)) < tol
        )

    def init_t(self, P: Tensor, V: Tensor) -> Tensor:
        # Initial guess is the intersection of rays with the X=0 or Y=O plane,
        # depending on if rays are mostly vertical or mostly horizontal
        init_x = -P[:, 0] / V[:, 0]
        init_y = -P[:, 1] / V[:, 1]
        init_t = torch.where(torch.abs(V[:, 0]) > torch.abs(V[:, 1]), init_x, init_y)

        return init_t

    def local_collide(self, P: Tensor, V: Tensor) -> tuple[Tensor, Tensor, Tensor]:

        dim = P.shape[1]

        init_t = self.init_t(P, V)

        t = self.collision_algorithm(self, P, V, init_t)

        local_points = P + t.unsqueeze(1).expand_as(V) * V

        if dim == 2:
            grad = self.f_grad(local_points)
        else:
            grad = self.F_grad(local_points)

        # Normalize gradient to make normal vectors
        local_normals = torch.nn.functional.normalize(grad, dim=1)

        # If there is no intersection, newton's method won't converge
        # and points will not be on the surface
        # So verify intersection here and filter points
        # that aren't on the surface
        # TODO test Newton method, and support tolerance configuration based on sampling dtype?
        valid = self.contains(local_points, tol=1e-3)

        return t, local_normals, valid

    def f(self, points: Tensor) -> Tensor:
        raise NotImplementedError

    def f_grad(self, points: Tensor) -> Tensor:
        raise NotImplementedError

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


class Parabola(ImplicitSurface):
    def __init__(
        self,
        diameter: float,
        a: int | float | nn.Parameter,
        **kwargs,
    ):
        super().__init__(CircularOutline(diameter), **kwargs)
        self.diameter = diameter
        self.a = to_tensor(a, default_dtype=self.dtype)

    def parameters(self) -> dict[str, nn.Parameter]:
        if isinstance(self.a, nn.Parameter):
            return {"a": self.a}
        else:
            return {}

    def samples2D(self, N: int) -> Tensor:
        """
        Generate N sample points located on the shape's curve with r >= 0
        """

        r = torch.linspace(0, self.outline.max_radius(), N)
        x = self.a * r**2
        return torch.stack((x, r), dim=-1)

    def extent_x(self) -> Tensor:
        r = self.outline.max_radius()
        return torch.as_tensor(self.a * r**2, dtype=self.dtype)

    def f(self, points: Tensor) -> Tensor:
        x, r = points[:, 0], points[:, 1]
        return torch.mul(self.a, torch.pow(r, 2)) - x

    def f_grad(self, points: Tensor) -> Tensor:
        x, r = points[:, 0], points[:, 1]
        return torch.stack((-torch.ones_like(x), 2 * self.a * r), dim=-1)

    def F(self, points: Tensor) -> Tensor:
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        return torch.mul(self.a, (y**2 + z**2)) - x

    def F_grad(self, points: Tensor) -> Tensor:
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        return torch.stack(
            (-torch.ones_like(x), 2 * self.a * y, 2 * self.a * z), dim=-1
        )


class Sphere(ImplicitSurface):
    """
    A section of a sphere, parameterized by curvature.
    Curvature is the inverse of radius: C = 1/R.

    This parameterization is useful because it enables clean representation of
    an infinite radius section of sphere (which is really a plane), and also
    enables changing the sign of C during optimization.

    In 2D, this surface is an arc of circle.
    In 3D, this surface is a section of a sphere (wikipedia call it a "spherical cap")

    For high curvature arcs (close to a half circle), it's better to use the
    Sphere2 class which uses radius parameterization and polar distance
    functions.
    """

    def __init__(self, diameter: float, r: int | float | nn.Parameter, **kwargs):
        super().__init__(outline=CircularOutline(diameter), **kwargs)
        self.diameter = diameter

        assert (
            torch.abs(torch.as_tensor(r)) >= diameter / 2
        ), f"Sphere diameter ({diameter}) must be less than 2x its arc radius (2x{r}={2*r})"

        self.K: torch.Tensor
        if isinstance(r, nn.Parameter):
            self.K = nn.Parameter(torch.tensor(1.0 / r.item(), dtype=self.dtype))
        else:
            self.K = torch.as_tensor(1.0 / r, dtype=self.dtype)

        assert self.K.dim() == 0

    def testname(self) -> str:
        R = 1 / self.K
        return f"Sphere-{self.diameter:.2f}-{R:.2f}"

    def parameters(self) -> dict[str, nn.Parameter]:
        if isinstance(self.K, nn.Parameter):
            return {"K": self.K}
        else:
            return {}

    def radius(self) -> Tensor:
        "Utility function because parameter is stored internally as curvature"
        return torch.div(1.0, self.K)

    def extent_x(self) -> Tensor:
        r = self.outline.max_radius()
        K = self.K
        return torch.div(K * r**2, 1 + torch.sqrt(1 - (r * K) ** 2))

    def samples2D(self, N: int, epsilon: float = 1e-3) -> Tensor:
        if self.K * self.diameter < 0.1:
            # If the curvature is low, use linear sampling along the y axis
            return sphere_samples_linear(
                curvature=self.K, start=0.0, end=self.outline.max_radius(), N=N
            )
        else:
            # Else, use the angular parameterization of the circle so that
            # samples are smoother, especially for high curvature circles.
            R = 1 / self.K
            theta_max = torch.arcsin(self.outline.max_radius() / torch.abs(R))
            return sphere_samples_angular(radius=R, start=0.0, end=theta_max, N=N)

    def samples2D_full(self, N: int, epsilon: float = 1e-3) -> Tensor:
        "Like samples2D but on the entire domain"
        if self.K * self.diameter < 0.1:
            # If the curvature is low, use linear sampling along the y axis
            return sphere_samples_linear(
                curvature=self.K,
                start=-self.outline.max_radius() + epsilon,
                end=self.outline.max_radius() - epsilon,
                N=N,
            )
        else:
            # Else, use the angular parameterization of the circle so that
            # samples are smoother, especially for high curvature circles.
            R = 1 / self.K
            theta_max = torch.arcsin(self.outline.max_radius() / torch.abs(R))
            return sphere_samples_angular(
                radius=R, start=-theta_max + epsilon, end=theta_max - epsilon, N=N
            )

    def f(self, points: Tensor) -> Tensor:
        x, r = points[:, 0], points[:, 1]
        r2 = torch.pow(r, 2)
        K = self.K

        # For points beyond the half-diameter
        # use the distance to the edge point
        center = torch.tensor([1 / K, 0.0])
        A = self.extent(dim=2, dtype=points.dtype) + torch.tensor(
            [0.0, self.diameter / 2]
        )
        B = self.extent(dim=2, dtype=points.dtype) - torch.tensor(
            [0.0, self.diameter / 2]
        )

        radicand = 1 - r2 * K**2
        safe_radicand = torch.clamp(radicand, min=0.0)
        circle = torch.div(K * r2, 1 + torch.sqrt(safe_radicand)) - x

        top_fallback = torch.linalg.vector_norm(points - A, dim=1)
        bottom_fallback = torch.linalg.vector_norm(points - B, dim=1)

        max_r = self.outline.max_radius()

        zone_mask = torch.abs(r) <= max_r

        return torch.where(
            zone_mask,
            circle,
            torch.where(r > 0, top_fallback, bottom_fallback),
        )

    def f_grad(self, points: Tensor) -> Tensor:
        x, r = points[:, 0], points[:, 1]
        r2 = torch.pow(r, 2)
        K = self.K

        # For points beyond the half-diameter
        # use the distance to the edge point
        center = torch.tensor([1 / K, 0.0])
        A = self.extent(dim=2, dtype=points.dtype) + torch.tensor(
            [0.0, self.diameter / 2]
        )
        B = self.extent(dim=2, dtype=points.dtype) - torch.tensor(
            [0.0, self.diameter / 2]
        )

        normA = torch.linalg.vector_norm(points - A, dim=1)
        normB = torch.linalg.vector_norm(points - B, dim=1)

        top_fallback = torch.stack(
            ((points[:, 0] - A[0]) / normA, (points[:, 1] - A[1]) / normA), dim=1
        )
        bottom_fallback = torch.stack(
            ((points[:, 0] - B[0]) / normB, (points[:, 1] - B[1]) / normB), dim=1
        )

        max_r = self.outline.max_radius()

        radicand = 1 - r2 * K**2

        # clamp to a non zero epsilon to avoid both sqrt(<0) and div by zero
        safe_radicand = torch.clamp(radicand, min=1e-4)
        grady = torch.div((K * r), torch.sqrt(safe_radicand))

        circle = torch.stack((-torch.ones_like(x), grady), dim=-1)

        assert circle.shape == top_fallback.shape == bottom_fallback.shape

        zone_mask = torch.abs(r) <= max_r

        return torch.where(
            zone_mask.unsqueeze(1).expand(-1, 2),
            circle,
            torch.where(
                r.unsqueeze(1).expand(-1, 2) > 0, top_fallback, bottom_fallback
            ),
        )

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


class Sphere3(ImplicitSurface):
    def __init__(
        self,
        diameter: float,
        r: int | float | nn.Parameter,
        **kwargs,
    ):
        super().__init__(outline=CircularOutline(diameter), **kwargs)
        self.diameter = diameter

        assert (
            torch.abs(torch.as_tensor(r)) >= diameter / 2
        ), f"Sphere diameter ({diameter}) must be less than 2x its arc radius (2x{r}={2*r})"

        self.K: torch.Tensor
        if isinstance(r, nn.Parameter):
            self.K = nn.Parameter(torch.tensor(1.0 / r.item(), dtype=self.dtype))
        else:
            self.K = torch.as_tensor(1.0 / r, dtype=self.dtype)

        assert self.K.dim() == 0

    def testname(self) -> str:
        R = 1 / self.K
        return f"Sphere3-{self.diameter:.2f}-{R:.2f}"

    def parameters(self) -> dict[str, nn.Parameter]:
        if isinstance(self.K, nn.Parameter):
            return {"K": self.K}
        else:
            return {}

    def radius(self) -> Tensor:
        "Utility function because parameter is stored internally as curvature"
        return torch.div(1.0, self.K)

    def extent_x(self) -> Tensor:
        r = self.outline.max_radius()
        K = self.K
        return torch.div(K * r**2, 1 + torch.sqrt(1 - (r * K) ** 2))

    def samples2D(self, N: int, epsilon: float = 1e-3) -> Tensor:
        if self.K * self.diameter < 0.1:
            # If the curvature is low, use linear sampling along the y axis
            return sphere_samples_linear(
                curvature=self.K, start=0.0, end=self.outline.max_radius(), N=N
            )
        else:
            # Else, use the angular parameterization of the circle so that
            # samples are smoother, especially for high curvature circles.
            R = 1 / self.K
            theta_max = torch.arcsin(self.outline.max_radius() / torch.abs(R))
            return sphere_samples_angular(radius=R, start=0.0, end=theta_max, N=N)

    def samples2D_full(self, N: int, epsilon: float = 1e-3) -> Tensor:
        "Like samples2D but on the entire domain"
        if self.K * self.diameter < 0.1:
            # If the curvature is low, use linear sampling along the y axis
            return sphere_samples_linear(
                curvature=self.K,
                start=-self.outline.max_radius() + epsilon,
                end=self.outline.max_radius() - epsilon,
                N=N,
            )
        else:
            # Else, use the angular parameterization of the circle so that
            # samples are smoother, especially for high curvature circles.
            R = 1 / self.K
            theta_max = torch.arcsin(self.outline.max_radius() / torch.abs(R))
            return sphere_samples_angular(
                radius=R, start=-theta_max + epsilon, end=theta_max - epsilon, N=N
            )

    def f(self, points: Tensor) -> Tensor:
        x, r = points[:, 0], points[:, 1]
        r2 = torch.pow(r, 2)
        K = self.K
        R = self.radius()
        center = torch.tensor([1 / K, 0.0])
        A = self.extent(dim=2, dtype=points.dtype) + torch.tensor(
            [0.0, self.diameter / 2]
        )
        B = self.extent(dim=2, dtype=points.dtype) - torch.tensor(
            [0.0, self.diameter / 2]
        )

        # normal vectors to the sector lines
        def norm(v):
            return torch.stack((v[1], -v[0]), dim=-1)

        LA = norm(A - center)
        LB = norm(center - B)
        center_vect = center - points

        zone_cone = torch.logical_and(
            torch.sum(LA * center_vect, dim=1) > 0,
            torch.sum(LB * center_vect, dim=1) > 0,
        )

        zone_A = torch.logical_and(torch.sum(LA * center_vect, dim=1) < 0, r > 0)

        full_circle = torch.linalg.norm(points - center, dim=1) - R
        dist_A = torch.linalg.norm(points - A, dim=1)
        dist_B = torch.linalg.norm(points - B, dim=1)

        return torch.where(zone_cone, full_circle, torch.where(zone_A, dist_A, dist_B))

    def f_grad(self, points: Tensor) -> Tensor:
        x, r = points[:, 0], points[:, 1]
        r2 = torch.pow(r, 2)
        K = self.K
        R = self.radius()
        center = torch.tensor([1 / K, 0.0])
        A = self.extent(dim=2, dtype=points.dtype) + torch.tensor(
            [0.0, self.diameter / 2]
        )
        B = self.extent(dim=2, dtype=points.dtype) - torch.tensor(
            [0.0, self.diameter / 2]
        )

        # normal vectors to the sector lines
        def norm(v):
            return torch.stack((v[1], -v[0]), dim=-1)

        LA = norm(A - center)
        LB = norm(center - B)
        center_vect = center - points

        zone_cone = torch.logical_and(
            torch.sum(LA * center_vect, dim=1) > 0,
            torch.sum(LB * center_vect, dim=1) > 0,
        )

        zone_A = torch.logical_and(torch.sum(LA * center_vect, dim=1) < 0, r > 0)

        dist_center = (points - center) / torch.linalg.norm(
            points - center, dim=1
        ).unsqueeze(1)
        full_circle = dist_center  # abs?
        dist_A = (points - A) / torch.linalg.norm(points - A, dim=1).unsqueeze(1)
        dist_B = (points - B) / torch.linalg.norm(points - B, dim=1).unsqueeze(1)

        return torch.where(
            zone_cone.unsqueeze(1),
            full_circle,
            torch.where(zone_A.unsqueeze(1), dist_A, dist_B),
        )

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


class Asphere(ImplicitSurface):
    def __init__(
        self,
        diameter: float,
        R: int | float | nn.Parameter,
        K: int | float | nn.Parameter,
        A4: int | float | nn.Parameter,
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__(CircularOutline(diameter), dtype)
        self.diameter = diameter

        self.C: torch.Tensor
        if isinstance(R, nn.Parameter):
            self.C = nn.Parameter(torch.tensor(1.0 / R.item(), dtype=dtype))
        else:
            self.C = torch.as_tensor(1.0 / R, dtype=dtype)
        assert self.C.dim() == 0

        self.K = to_tensor(K)
        self.A4 = to_tensor(A4)

    def parameters(self) -> dict[str, nn.Parameter]:
        possible = {
            "C": self.C,  # curvature
            "K": self.K,
            "A4": self.A4,
        }
        return {
            name: value
            for name, value in possible.items()
            if isinstance(value, nn.Parameter)
        }

    def extent_x(self) -> Tensor:
        r2 = self.outline.max_radius() ** 2
        C, K, A4 = self.C, self.K, self.A4
        C2 = torch.pow(C, 2)
        return torch.div(C * r2, 1 + torch.sqrt(1 - (1 + K) * r2 * C2)) + A4 * r2**2

    def samples2D(self, N: int) -> Tensor:
        K, C, A4 = self.K, self.C, self.A4
        C2 = torch.pow(C, 2)

        Y = torch.linspace(0, 1, N) * self.outline.max_radius()
        r2 = torch.pow(Y, 2)
        X = torch.div(C * r2, 1 + torch.sqrt(1 - (1 + K) * r2 * C2)) + A4 * torch.pow(
            r2, 2
        )

        return torch.stack((X, Y), dim=-1)

    def f(self, points: Tensor) -> Tensor:
        x, r = points[:, 0], points[:, 1]
        r2 = torch.pow(r, 2)
        K, C, A4 = self.K, self.C, self.A4
        C2 = torch.pow(C, 2)
        return (
            torch.div(C * r2, 1 + torch.sqrt(1 - (1 + K) * r2 * C2))
            + A4 * torch.pow(r2, 2)
            - x
        )

    def f_grad(self, points: Tensor) -> Tensor:
        x, r = points[:, 0], points[:, 1]
        r2 = torch.pow(r, 2)
        K, C, A4 = self.K, self.C, self.A4
        C2 = torch.pow(C, 2)

        xgrad = -torch.ones_like(x)
        rgrad = torch.div(
            C * r, torch.sqrt(1 - (1 + K) * r2 * C2)
        ) + 4 * A4 * torch.pow(r, 3)

        return torch.stack((xgrad, rgrad), dim=-1)

    def F(self, points: Tensor) -> Tensor:
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        K, C, A4 = self.K, self.C, self.A4
        C2 = torch.pow(C, 2)
        r2 = y**2 + z**2

        return (
            torch.div(C * r2, 1 + torch.sqrt(1 - (1 + K) * r2 * C2))
            + A4 * torch.pow(r2, 2)
            - x
        )

    def F_grad(self, points: Tensor) -> Tensor:
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        K, C, A4 = self.K, self.C, self.A4
        C2 = torch.pow(C, 2)
        r2 = y**2 + z**2

        denom = torch.sqrt(1 - (1 + K) * r2 * C2)
        coeffs_term = 4 * A4 * r2

        return torch.stack(
            (
                -torch.ones_like(x),
                (C * y) / denom + y * coeffs_term,
                (C * z) / denom + z * coeffs_term,
            ),
            dim=-1,
        )
