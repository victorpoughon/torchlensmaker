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

from torchlensmaker.core.collision_detection import CollisionMethod, default_collision_method

# shorter for type annotations
Tensor = torch.Tensor


class LocalSurface:
    """
    Base class for surfaces defined in a local reference frame
    """

    def __init__(self, outline: Outline, dtype: torch.dtype = torch.float64):
        self.outline = outline
        self.dtype = dtype

    def testname(self) -> str:
        "A string for identification in test cases"
        raise NotImplementedError

    def parameters(self) -> dict[str, nn.Parameter]:
        raise NotImplementedError
    
    def zero(self, dim: int, dtype: torch.dtype) -> Tensor:
        "N-dimensional zero point"
        return torch.zeros((dim,), dtype=dtype)
    
    def extent(self, dim: int, dtype: torch.dtype) -> Tensor:
        "N-dimensional extent point"
        return torch.cat(
            (self.extent_x().unsqueeze(0), torch.zeros(dim - 1, dtype=dtype)),
            dim=0,
        )
    
    def extent_x(self) -> Tensor:
        """
        Extent along the X axis
        i.e. X coordinate of the point on the surface that is furthest along the X axis
        """
        raise NotImplementedError
    
    def normals(self, points: Tensor) -> Tensor:
        """
        Unit vectors normal to the surface at input points of shape (N, D)
        """
        raise NotImplementedError
    
    def contains(self, points: Tensor, tol: float = 1e-6) -> Tensor:
        raise NotImplementedError
    
    def samples2D_half(self, N: int, epsilon: float = 1e-3) -> Tensor:
        "Generate 2D samples on the half positive domain"
        raise NotImplementedError
    
    def samples2D_full(self, N: int, epsilon: float = 1e-3) -> Tensor:
        "Generate 2D samples on the full domain"
        raise NotImplementedError

    def local_collide(self, P: Tensor, V: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Find collision points and surface normals of ray-surface intersection
        for parametric rays P+tV expressed in the surface local frame.

        Returns:
            t: Value of parameter t such that P + tV is on the surface
            normals: Normal unit vectors to the surface at the collision points
            valid: Bool tensor indicating which rays do collide with the surface
        """
        raise NotImplementedError


class Plane(LocalSurface):
    "X=0 plane"

    def __init__(self, outline: Outline, dtype: torch.dtype):
        super().__init__(outline, dtype)

    def parameters(self) -> dict[str, nn.Parameter]:
        return {}

    def samples2D_half(self, N: int, epsilon: float = 1e-3) -> Tensor:
        r = torch.linspace(0, self.outline.max_radius(), N)
        return torch.stack((torch.zeros(N), r), dim=-1)

    def samples2D_full(self, N: int, epsilon: float = 1e-3) -> Tensor:
        maxr = self.outline.max_radius()
        r = torch.linspace(-maxr, maxr, N)
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

    def normals(self, points: Tensor) -> Tensor:
        N, dim = points.shape
        normal = (
            torch.tensor([-1.0, 0.0], dtype=self.dtype)
            if dim == 2
            else torch.tensor([-1.0, 0.0, 0.0], dtype=self.dtype)
        )
        return torch.tile(normal, (N, 1))

    def extent_x(self) -> Tensor:
        return torch.as_tensor(0.0, dtype=self.dtype)

    def contains(self, points: Tensor, tol: float = 1e-6) -> Tensor:
        return torch.logical_and(
            self.outline.contains(points), torch.abs(points[:, 0]) < tol
        )


class SquarePlane(Plane):
    def __init__(self, side_length: float, dtype: torch.dtype = torch.float64):
        self.side_length = side_length
        super().__init__(SquareOutline(side_length), dtype)

    def testname(self) -> str:
        return f"SquarePlane-{self.side_length}"


class CircularPlane(Plane):
    "aka disk"

    def __init__(self, diameter: float, dtype: torch.dtype = torch.float64):
        self.diameter = diameter
        super().__init__(CircularOutline(diameter), dtype)

    def testname(self) -> str:
        return f"CircularPlane-{self.diameter}"


class ImplicitSurface(LocalSurface):
    """
    Surface3D defined in implicit form: F(x,y,z) = 0
    """

    def __init__(
        self,
        collision_method: CollisionMethod = default_collision_method,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.collision_method = collision_method

    def contains(self, points: Tensor, tol: float = 1e-6) -> Tensor:
        dim = points.shape[1]

        F = self.F if dim == 3 else self.f

        return torch.logical_and(
            self.outline.contains(points), torch.abs(F(points)) < tol
        )

    def local_collide(self, P: Tensor, V: Tensor) -> tuple[Tensor, Tensor, Tensor]:

        t = self.collision_method(self, P, V)

        local_points = P + t.unsqueeze(1).expand_as(V) * V
        local_normals = self.normals(local_points)

        # If there is no intersection, collision detection won't converge
        # and points will not be on the surface
        # So verify intersection here and filter points
        # that aren't on the surface
        # TODO better tolerance configuration based on sampling dtype
        valid = self.contains(local_points, tol=1e-3)

        return t, local_normals, valid

    def normals(self, points: Tensor) -> Tensor:
        return nn.functional.normalize(self.Fd_grad(points), dim=1)

    def Fd(self, points: Tensor) -> Tensor:
        "Calls f or F depending on the shape of points"
        return self.f(points) if points.shape[1] == 2 else self.F(points)

    def Fd_grad(self, points: Tensor) -> Tensor:
        "Calls f_grad or F_grad depending on the shape of points"
        return self.f_grad(points) if points.shape[1] == 2 else self.F_grad(points)

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
        super().__init__(outline=CircularOutline(diameter), **kwargs)
        self.diameter = diameter
        self.a = to_tensor(a, default_dtype=self.dtype)
    
    def testname(self) -> str:
        return f"Parabola-{self.diameter:.2f}-{self.a.item():.2f}"

    def parameters(self) -> dict[str, nn.Parameter]:
        if isinstance(self.a, nn.Parameter):
            return {"a": self.a}
        else:
            return {}

    def samples2D_half(self, N: int, epsilon: float = 1e-3) -> Tensor:
        r = torch.linspace(0, self.outline.max_radius(), N)
        x = self.a * r**2
        return torch.stack((x, r), dim=-1)
    
    def samples2D_full(self, N: int, epsilon: float = 1e-3) -> Tensor:
        r = torch.linspace(-self.outline.max_radius(), self.outline.max_radius(), N)
        x = self.a * r**2
        return torch.stack((x, r), dim=-1)

    def extent_x(self) -> Tensor:
        r = self.outline.max_radius()
        return torch.as_tensor(self.a * r**2, dtype=self.dtype)

    # TODO add zone band mask to parabola
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
    SphereR class which uses radius parameterization and polar distance
    functions.
    """

    def __init__(
        self,
        diameter: float,
        R: int | float | nn.Parameter | None = None,
        C: int | float | nn.Parameter | None = None,
        **kwargs,
    ):
        super().__init__(outline=CircularOutline(diameter), **kwargs)
        self.diameter = diameter

        if (R is not None and C is not None) or (R is None and C is None):
            raise RuntimeError(
                "Sphere must be initialized with exactly one of R (radius) or C (curvature)."
            )

        self.C: torch.Tensor
        if C is None:
            if torch.abs(torch.as_tensor(R)) < diameter / 2:
                raise RuntimeError(
                    f"Sphere radius (R={R}) must be at least half the surface diameter (D={diameter})"
                )

            if isinstance(R, nn.Parameter):
                self.C = nn.Parameter(torch.tensor(1.0 / R.item(), dtype=self.dtype))
            else:
                self.C = torch.as_tensor(1.0 / R, dtype=self.dtype)
        else:
            if isinstance(C, nn.Parameter):
                self.C = C
            else:
                self.C = torch.as_tensor(C, dtype=self.dtype)

        assert self.C.dim() == 0

    def testname(self) -> str:
        return f"Sphere-{self.diameter:.2f}-{self.C.item():.2f}"

    def parameters(self) -> dict[str, nn.Parameter]:
        if isinstance(self.C, nn.Parameter):
            return {"C": self.C}
        else:
            return {}
    
    def radius(self) -> float:
        return 1 / self.C.item()

    def extent_x(self) -> Tensor:
        r = self.outline.max_radius()
        C = self.C
        return torch.div(C * r**2, 1 + torch.sqrt(1 - (r * C) ** 2))

    def samples2D_half(self, N: int, epsilon: float = 1e-3) -> Tensor:
        # If the curvature is low, use linear sampling along the y axis
        # Else, use the angular parameterization of the circle so that
        # samples are smoother, especially for high curvature circles.
        if self.C * self.diameter < 0.1:
            return sphere_samples_linear(
                curvature=self.C,
                start=0.0,
                end=self.outline.max_radius() - epsilon,
                N=N,
            )
        else:
            R = 1 / self.C
            theta_max = torch.arcsin(self.outline.max_radius() / torch.abs(R))
            return sphere_samples_angular(
                radius=R, start=0.0, end=theta_max - epsilon, N=N
            )

    def samples2D_full(self, N: int, epsilon: float = 1e-3) -> Tensor:
        # If the curvature is low, use linear sampling along the y axis
        # Else, use the angular parameterization of the circle so that
        # samples are smoother, especially for high curvature circles.
        if self.C * self.diameter < 0.1:
            return sphere_samples_linear(
                curvature=self.C,
                start=-self.outline.max_radius() + epsilon,
                end=self.outline.max_radius() - epsilon,
                N=N,
            )
        else:
            R = 1 / self.C
            theta_max = torch.arcsin(self.outline.max_radius() / torch.abs(R))
            return sphere_samples_angular(
                radius=R, start=-theta_max + epsilon, end=theta_max - epsilon, N=N
            )

    def edge_points(self, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        "2D edge points"

        A = self.extent(dim=2, dtype=dtype) + torch.tensor([0.0, self.diameter / 2])
        B = self.extent(dim=2, dtype=dtype) - torch.tensor([0.0, self.diameter / 2])
        return A, B

    def f(self, points: Tensor) -> Tensor:
        x, r = points[:, 0], points[:, 1]
        r2 = torch.pow(r, 2)
        C = self.C

        # For points beyond the half-diameter
        # use the distance to the edge point
        A, B = self.edge_points(dtype=points.dtype)

        radicand = 1 - r2 * C**2
        safe_radicand = torch.clamp(radicand, min=0.0)
        circle = torch.div(C * r2, 1 + torch.sqrt(safe_radicand)) - x

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
        C = self.C

        # For points beyond the half-diameter
        # use the distance to the edge point
        A, B = self.edge_points(dtype=points.dtype)

        normA = torch.linalg.vector_norm(points - A, dim=1)
        normB = torch.linalg.vector_norm(points - B, dim=1)

        top_fallback = torch.stack(
            ((points[:, 0] - A[0]) / normA, (points[:, 1] - A[1]) / normA), dim=1
        )
        bottom_fallback = torch.stack(
            ((points[:, 0] - B[0]) / normB, (points[:, 1] - B[1]) / normB), dim=1
        )

        max_r = self.outline.max_radius()

        radicand = 1 - r2 * C**2

        # clamp to a non zero epsilon to avoid both sqrt(<0) and div by zero
        safe_radicand = torch.clamp(radicand, min=1e-4)
        grady = torch.div((C * r), torch.sqrt(safe_radicand))

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

    # TODO zone mask in 3D
    def F(self, points: Tensor) -> Tensor:
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        C = self.C
        r2 = y**2 + z**2
        return torch.div(C * r2, 1 + torch.sqrt(1 - r2 * C**2)) - x

    def F_grad(self, points: Tensor) -> Tensor:
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        C = self.C
        r2 = y**2 + z**2
        denom = torch.sqrt(1 - r2 * C**2)
        return torch.stack(
            (-torch.ones_like(x), (C * y) / denom, (C * z) / denom), dim=-1
        )


class SphereR(LocalSurface):
    """
    A section of a sphere, parameterized by signed radius.

    This parameterization is useful to represent high curvature sections
    including a complete half-sphere. However it's poorly suited to represent
    low curvature sections that are closer to a planar surface.

    In 2D, this surface is an arc of circle.
    In 3D, this surface is a section of a sphere (wikipedia call it a "spherical cap")
    """

    def __init__(
        self,
        diameter: float,
        R: int | float | nn.Parameter | None = None,
        C: int | float | nn.Parameter | None = None,
        **kwargs,
    ):
        super().__init__(outline=CircularOutline(diameter), **kwargs)
        self.diameter = diameter

        if (R is not None and C is not None) or (R is None and C is None):
            raise RuntimeError(
                "SphereR must be initialized with exactly one of R (radius) or C (curvature)."
            )

        self.R: torch.Tensor
        if C is None:
            if torch.abs(torch.as_tensor(R)) < diameter / 2:
                raise RuntimeError(
                    f"Sphere radius (R={R}) must be at least half the surface diameter (D={diameter})"
                )

            if isinstance(R, nn.Parameter):
                self.R = R
            else:
                self.R = torch.as_tensor(R, dtype=self.dtype)
        else:
            if isinstance(C, nn.Parameter):
                self.R = nn.Parameter(torch.tensor(1.0 / C.item(), dtype=self.dtype))
            else:
                self.R = torch.as_tensor(1.0 / C, dtype=self.dtype)

        assert self.R.dim() == 0

    def testname(self) -> str:
        return f"SphereR-{self.diameter:.2f}-{self.R.item():.2f}"

    def parameters(self) -> dict[str, nn.Parameter]:
        if isinstance(self.R, nn.Parameter):
            return {"R": self.R}
        else:
            return {}
    
    def radius(self) -> float:
        return self.R.item()

    def extent_x(self) -> Tensor:
        r = self.outline.max_radius()
        K = 1 / self.R
        return torch.div(K * r**2, 1 + torch.sqrt(1 - (r * K) ** 2))

    def samples2D_half(self, N: int, epsilon: float = 1e-3) -> Tensor:
        C = 1 / self.R
        if torch.abs(C * self.diameter) < 0.1:
            # If the curvature is low, use linear sampling along the y axis
            return sphere_samples_linear(
                curvature=C, start=0.0, end=self.outline.max_radius(), N=N
            )
        else:
            # Else, use the angular parameterization of the circle so that
            # samples are smoother, especially for high curvature circles.
            theta_max = torch.arcsin(self.outline.max_radius() / torch.abs(self.R))
            return sphere_samples_angular(radius=self.R, start=0.0, end=theta_max, N=N)

    def samples2D_full(self, N: int, epsilon: float = 1e-3) -> Tensor:
        "Like samples2D but on the entire domain"
        C = 1 / self.R
        if torch.abs(C * self.diameter) < 0.1:
            # If the curvature is low, use linear sampling along the y axis
            return sphere_samples_linear(
                curvature=C,
                start=-self.outline.max_radius() + epsilon,
                end=self.outline.max_radius() - epsilon,
                N=N,
            )
        else:
            # Else, use the angular parameterization of the circle so that
            # samples are smoother, especially for high curvature circles.
            theta_max = torch.arcsin(self.outline.max_radius() / torch.abs(self.R))
            return sphere_samples_angular(
                radius=self.R, start=-theta_max + epsilon, end=theta_max - epsilon, N=N
            )

    def center(self, dim: int) -> Tensor:
        if dim == 2:
            return torch.tensor([self.R, 0.0])
        else:
            return torch.tensor([self.R, 0.0, 0.0])

    def normals(self, points: Tensor) -> Tensor:
        # The normal is the vector from the center to the points
        center = self.center(dim=points.shape[1])
        return torch.nn.functional.normalize(points - center)

    def contains(self, points: Tensor, tol: float = 1e-6) -> Tensor:
        center = self.center(dim=points.shape[1])
        within_outline = self.outline.contains(points)
        within_sphere = torch.abs(torch.linalg.vector_norm(points - center, dim=1) - torch.abs(self.R)) <= tol
        within_extent = torch.abs(points[:, 0]) <= torch.abs(self.extent_x())

        return torch.all(torch.stack((within_outline, within_sphere, within_extent), dim=1), dim=1)


    def local_collide(self, P: Tensor, V: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        N, D = P.shape

        # Sphere-ray collision is a second order polynomial
        center = self.center(dim=D)
        A = torch.sum(V**2, dim=1)
        B = 2 * torch.sum(V * (P - center), dim=1)
        C = torch.sum((P - center) ** 2, dim=1) - self.R**2
        assert A.shape == B.shape == C.shape == (N,)

        delta = B**2 - 4 * A * C
        safe_delta = torch.clamp(delta, min=0.0)
        assert delta.shape == (N,), delta.shape
        assert safe_delta.shape == (N,), safe_delta.shape

        # tensor of shape (N, 2) with both roots
        # safe meaning that if the root is undefined the value is zero instead
        safe_roots = torch.stack(
            (
                (-B + torch.sqrt(safe_delta)) / (2 * A),
                (-B - torch.sqrt(safe_delta)) / (2 * A),
            ),
            dim=1,
        )
        assert safe_roots.shape == (N, 2)

        # mask of shape (N, 2) indicating if each root is inside the outline
        root_inside = torch.stack(
            (
                self.contains(P + safe_roots[:, 0].unsqueeze(1).expand_as(V) * V),
                self.contains(P + safe_roots[:, 1].unsqueeze(1).expand_as(V) * V),
            ),
            dim=1,
        )
        assert root_inside.shape == (N, 2)

        # number of valid roots
        number_of_valid_roots = torch.sum(root_inside, dim=1)
        assert number_of_valid_roots.shape == (N,)

        # index of the first valid root
        _, index_first_valid = torch.max(root_inside, dim=1)
        assert index_first_valid.shape == (N,)

        # index of the root closest to zero
        _, index_closest = torch.min(torch.abs(safe_roots), dim=1)
        assert index_closest.shape == (N,)

        # delta < 0 => no collision
        # delta >=0 => two roots (which maybe equal)
        #  - if both are outside the outline => no collision
        #  - if only one is inside the outline => one collision
        #  - if both are inside the outline => return the root closest to zero (i.e. the ray origin)

        default_t = torch.zeros(N, dtype=self.dtype)
        arange = torch.arange(N)
        t = torch.where(
            delta < 0,
            default_t,
            torch.where(
                number_of_valid_roots == 0,
                default_t,
                torch.where(
                    number_of_valid_roots == 1,
                    safe_roots[arange, index_first_valid],
                    torch.where(
                        number_of_valid_roots == 2,
                        safe_roots[arange, index_closest],
                        default_t,
                    ),
                ),
            ),
        )

        local_points = P + t.unsqueeze(1).expand_as(V) * V
        local_normals = self.normals(local_points)
        valid = self.contains(local_points)

        return t, local_normals, valid


class Asphere(ImplicitSurface):
    def __init__(
        self,
        diameter: float,
        R: int | float | nn.Parameter,
        K: int | float | nn.Parameter,
        A4: int | float | nn.Parameter,
        **kwargs
    ):
        super().__init__(outline=CircularOutline(diameter), **kwargs)
        self.diameter = diameter

        self.C: torch.Tensor
        if isinstance(R, nn.Parameter):
            self.C = nn.Parameter(torch.tensor(1.0 / R.item(), dtype=self.dtype))
        else:
            self.C = torch.as_tensor(1.0 / R, dtype=self.dtype)
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

    def samples2D_half(self, N: int) -> Tensor:
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
