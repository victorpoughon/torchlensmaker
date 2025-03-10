import torch
import torch.nn as nn
import math

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

from torchlensmaker.core.collision_detection import (
    CollisionMethod,
    default_collision_method,
)
from torchlensmaker.core.geometry import unit_vector, within_radius
from torchlensmaker.core.collision_detection import init_closest_origin

from torch.linalg import vector_norm

from typing import Any, Optional, Type

# shorter for type annotations
Tensor = torch.Tensor


class LocalSurface:
    """
    Base class for surfaces
    """

    def __init__(self, dtype: torch.dtype = torch.float64):
        self.dtype = dtype

    def testname(self) -> str:
        "A string for identification in test cases"
        raise NotImplementedError

    def parameters(self) -> dict[str, nn.Parameter]:
        raise NotImplementedError

    def zero(self, dim: int) -> Tensor:
        "N-dimensional zero point"
        return torch.zeros((dim,), dtype=self.dtype)

    def extent(self, dim: int) -> Tensor:
        "N-dimensional extent point"
        return torch.cat(
            (self.extent_x().unsqueeze(0), torch.zeros(dim - 1, dtype=self.dtype)),
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
        Unit vectors normal to the surface at input points of shape (..., D)
        All dimensions except the last one are batch dimensions
        Input points are not necessarily on the surface
        """
        raise NotImplementedError

    def contains(self, points: Tensor, tol: Optional[float] = None) -> Tensor:
        """
        Test if points belong to the surface

        Args:
            * points: Input points of shape (..., D)
            * tol: optional tolerance parameter (default None). None means use an internal default based on dtype

        Returns:
            * boolean mask of shape points.shape[:-1]
        """
        raise NotImplementedError

    def samples2D_half(self, N: int, epsilon: float) -> Tensor:
        "Generate 2D samples on the half positive domain"
        raise NotImplementedError

    def samples2D_full(self, N: int, epsilon: float) -> Tensor:
        "Generate 2D samples on the full domain"
        raise NotImplementedError

    def bounding_radius(self) -> float:
        """
        Any point on the surface has a distance to the center that is less
        than (or equal) to the bounding radius
        """
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
        super().__init__(dtype)
        self.outline = outline

    def parameters(self) -> dict[str, nn.Parameter]:
        return {}

    def samples2D_half(self, N: int, epsilon: float) -> Tensor:
        maxr = (1.0 - epsilon) * self.outline.max_radius()
        r = torch.linspace(0, maxr, N, dtype=self.dtype)
        return torch.stack((torch.zeros(N), r), dim=-1)

    def samples2D_full(self, N: int, epsilon: float) -> Tensor:
        maxr = (1.0 - epsilon) * self.outline.max_radius()
        r = torch.linspace(-maxr, maxr, N, dtype=self.dtype)
        return torch.stack((torch.zeros(N), r), dim=-1)

    def local_collide(self, P: Tensor, V: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        # Nominal case: rays are not perpendicular to the X axis
        mask_nominal = V[:, 0] != torch.zeros(1, dtype=V.dtype)

        t = torch.where(
            mask_nominal,
            -P[:, 0] / V[:, 0],  # Nominal case, rays aren't vertical
            init_closest_origin(self, P, V)  # Default for vertical rays
        )

        local_points = P + t.unsqueeze(1).expand_as(V) * V
        valid = self.outline.contains(local_points)
        return t, self.normals(local_points), valid

    def normals(self, points: Tensor) -> Tensor:
        batch, dim = points.shape[:-1], points.shape[-1]
        normal = -unit_vector(dim=dim, dtype=self.dtype)
        return torch.tile(normal, (*batch, 1))

    def extent_x(self) -> Tensor:
        return torch.as_tensor(0.0, dtype=self.dtype)

    def bounding_radius(self) -> float:
        return self.outline.max_radius()

    def contains(self, points: Tensor, tol: Optional[float] = None) -> Tensor:
        if tol is None:
            tol = {torch.float32: 1e-4, torch.float64: 1e-6}[self.dtype]

        return torch.logical_and(
            self.outline.contains(points, tol), torch.abs(points.select(-1, 0)) < tol
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
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__(dtype=dtype)
        self.collision_method = collision_method

    def contains(self, points: Tensor, tol: Optional[float] = None) -> Tensor:

        if tol is None:
            tol = {torch.float32: 1e-4, torch.float64: 1e-7}[self.dtype]

        dim = points.shape[1]

        F = self.F if dim == 3 else self.f

        return torch.abs(F(points)) < tol

    def rmse(self, points: Tensor) -> float:
        N = sum(points.shape[:-1])
        return torch.sqrt(torch.sum(self.Fd(points) ** 2) / N).item()

    def local_collide(self, P: Tensor, V: Tensor) -> tuple[Tensor, Tensor, Tensor]:

        t = self.collision_method(self, P, V, history=False).t

        local_points = P + t.unsqueeze(-1).expand_as(V) * V
        local_normals = self.normals(local_points)

        # If there is no intersection, collision detection won't converge
        # and points will not be on the surface
        # So verify intersection here and filter points
        # that aren't on the surface
        valid = self.contains(local_points)

        return t, local_normals, valid

    def normals(self, points: Tensor) -> Tensor:
        # To get the normals of an implicit surface,
        # normalize the gradient of the implicit function
        return nn.functional.normalize(self.Fd_grad(points), dim=-1)

    def Fd(self, points: Tensor) -> Tensor:
        "Calls f or F depending on the shape of points"
        return self.f(points) if points.shape[-1] == 2 else self.F(points)

    def Fd_grad(self, points: Tensor) -> Tensor:
        "Calls f_grad or F_grad depending on the shape of points"
        return self.f_grad(points) if points.shape[-1] == 2 else self.F_grad(points)

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


class DiameterBandSurfaceSq(ImplicitSurface):
    "Distance to edge points"

    def __init__(self, Ax: Tensor, Ar: Tensor, dtype: torch.dtype = torch.float64):
        super().__init__(dtype=dtype)
        self.Ax = Ax
        self.Ar = Ar

    def f(self, points: Tensor) -> Tensor:
        assert points.shape[-1] == 2
        X, R = points.unbind(-1)
        Ax, Ar = self.Ax, self.Ar
        return torch.sqrt((X - Ax) ** 2 + (torch.abs(R) - Ar) ** 2)

    def f_grad(self, points: Tensor) -> Tensor:
        assert points.shape[-1] == 2
        X, R = points.unbind(-1)
        Ax, Ar = self.Ax, self.Ar
        sq = self.f(points)
        return torch.stack(
            ((X - Ax) / sq, torch.sign(R) * (torch.abs(R) - Ar) / sq), dim=-1
        )

    def F(self, points: Tensor) -> Tensor:
        assert points.shape[-1] == 3
        X, Y, Z = points.unbind(-1)
        R2 = Y**2 + Z**2
        Ax, Ar = self.Ax, self.Ar
        return torch.sqrt((X - Ax) ** 2 + (torch.sqrt(R2) - Ar) ** 2)

    def F_grad(self, points: Tensor) -> Tensor:
        assert points.shape[-1] == 3
        X, Y, Z = points.unbind(-1)
        R2 = Y**2 + Z**2
        Ax, Ar = self.Ax, self.Ar
        sq = self.F(points)
        sqr2 = torch.sqrt(R2)
        quot = (sqr2 - Ar) / (sqr2 * sq)
        return torch.stack(
            (
                (X - Ax) / sq,
                Y * quot,
                Z * quot,
            ),
            dim=-1,
        )


class SagSurface(ImplicitSurface):
    """
    Axially symmetric implicit surface defined by a sag function.

    A sag function g(r) is a one dimensional real valued function that describes
    a surface x coordinate in an arbitrary meridional plane (x,r) as a function
    of the distance to the principal axis: x = g(r).

    Derived classes provide the sag function and its gradient in
    both 2 and 3 dimensions. This class then uses it to create the implicit
    function F representing the corresponding implicit surface.

    The sag function is assumed defined only on the domain (- diameter/2 ;
    diameter / 2) in the meridional plane. Outside of this domain, a fallback
    function is used (implemented by DiameterBandSurfaceSq).
    """

    def __init__(
        self,
        diameter: float,
        collision_method: CollisionMethod = default_collision_method,
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__(collision_method=collision_method, dtype=dtype)
        self.diameter = diameter

    def mask_function(self, points: Tensor) -> Tensor:
        return within_radius(self.diameter / 2, points)

    def fallback_surface(self) -> DiameterBandSurfaceSq:
        return DiameterBandSurfaceSq(
            Ax=self.extent_x(),
            Ar=torch.as_tensor(self.diameter / 2, dtype=self.dtype),
            dtype=self.dtype,
        )

    def bounding_radius(self) -> float:
        """
        Any point on the surface has a distance to the center that is less
        than (or equal) to the bounding radius
        """
        return math.sqrt((self.diameter / 2) ** 2 + self.extent_x() ** 2)

    def g(self, r: Tensor) -> Tensor:
        """
        2D sag function $g(r)$

        Args:
        * r: batched tensor of shape (...)

        Returns:
        * batched tensor of shape (...)
        """
        raise NotImplementedError

    def g_grad(self, r: Tensor) -> Tensor:
        """
        Derivative of the 2D sag function $g'(r)$

        Args:
        * r: batched tensor of shape (...)

        Returns:
        * batched tensor of shape (...)
        """
        raise NotImplementedError

    def G(self, y: Tensor, z: Tensor) -> Tensor:
        """
        3D sag function $G(X, Y) = g(\\sqrt{y^2 + z^2})$

        Args:
        * y: batched tensor of shape (...)
        * z: batched tensor of shape (...)

        Returns:
        * batched tensor of shape (...)
        """
        raise NotImplementedError

    def G_grad(self, y: Tensor, z: Tensor) -> tuple[Tensor, Tensor]:
        """
        Gradient of the 3D sag function $\\nabla G(y, z)$

        Args:
        * y: batched tensor of shape (...)
        * z: batched tensor of shape (...)

        Returns:
        * grad_y: batched tensor of shape (...)
        * grad_z: batched tensor of shape (...)
        """
        raise NotImplementedError

    def f(self, points: Tensor) -> Tensor:
        assert points.shape[-1] == 2
        x, r = points.unbind(-1)
        sag_f = self.g(r) - x
        mask = self.mask_function(points)
        fallback = self.fallback_surface()
        return torch.where(mask, sag_f, fallback.f(points))

    def f_grad(self, points: Tensor) -> Tensor:
        assert points.shape[-1] == 2
        x, r = points.unbind(-1)
        sag_f_grad = torch.stack((-torch.ones_like(x), self.g_grad(r)), dim=-1)
        mask = self.mask_function(points)
        fallback = self.fallback_surface()
        return torch.where(
            mask.unsqueeze(-1).expand(*mask.size(), 2),
            sag_f_grad,
            fallback.f_grad(points),
        )

    def F(self, points: Tensor) -> Tensor:
        assert points.shape[-1] == 3
        x, y, z = points.unbind(-1)
        sag_F = self.G(y, z) - x
        mask = self.mask_function(points)
        fallback = self.fallback_surface()
        return torch.where(mask, sag_F, fallback.F(points))

    def F_grad(self, points: Tensor) -> Tensor:
        assert points.shape[-1] == 3
        x, y, z = points.unbind(-1)
        grad_y, grad_z = self.G_grad(y, z)
        sag_F_grad = torch.stack((-torch.ones_like(x), grad_y, grad_z), dim=-1)
        mask = self.mask_function(points)
        fallback = self.fallback_surface()
        return torch.where(
            mask.unsqueeze(-1).expand(*mask.size(), 3),
            sag_F_grad,
            fallback.F_grad(points),
        )


class Sphere(SagSurface):
    """
    A section of a sphere, parameterized by signed curvature.
    Curvature is the inverse of radius: C = 1/R.

    This parameterization is useful because it enables clean representation of
    an infinite radius section of sphere (which is really a plane), and also
    enables changing the sign of C during optimization.

    In 2D, this surface is an arc of circle.
    In 3D, this surface is a section of a sphere (wikipedia calls it a "spherical cap")

    For high curvature arcs (close to a half circle), it's better to use the
    SphereR class which uses radius parameterization and polar distance
    functions. In fact this class cannot represent an exact half circle (R =
    D/2) due to the gradient becoming infinite, use SphereR instead.
    """

    def __init__(
        self,
        diameter: float,
        R: int | float | nn.Parameter | None = None,
        C: int | float | nn.Parameter | None = None,
        collision_method: CollisionMethod = default_collision_method,
        dtype: torch.dtype = torch.float64,
    ):
        if (R is not None and C is not None) or (R is None and C is None):
            raise RuntimeError(
                "Sphere must be initialized with exactly one of R (radius) or C (curvature)."
            )

        C_tensor: torch.Tensor | nn.Parameter
        if C is None and R is not None:
            if torch.abs(torch.as_tensor(R)) <= diameter / 2:
                raise RuntimeError(
                    f"Sphere radius (R={R}) must be strictly greater than half the surface diameter (D/2={diameter/2}) "
                    f"(To model an exact half-sphere, use SphereR)."
                )

            if isinstance(R, nn.Parameter):
                C_tensor = nn.Parameter(torch.tensor(1.0 / R.item(), dtype=R.dtype))
            else:
                C_tensor = torch.as_tensor(1.0 / R, dtype=dtype)
        else:
            if isinstance(C, nn.Parameter):
                C_tensor = C
            else:
                C_tensor = torch.as_tensor(C, dtype=dtype)

        assert C_tensor.dim() == 0
        assert C_tensor.dtype == dtype
        self.C = C_tensor

        super().__init__(
            diameter=diameter, collision_method=collision_method, dtype=dtype
        )

    def radius(self) -> float:
        "Utility function to get radius from internal curvature"
        return torch.div(torch.tensor(1.0, dtype=self.dtype), self.C).item()

    def testname(self) -> str:
        return f"Sphere-{self.diameter:.2f}-{self.C.item():.2f}"

    def parameters(self) -> dict[str, nn.Parameter]:
        return {"C": self.C} if isinstance(self.C, nn.Parameter) else {}

    def extent_x(self) -> Tensor:
        r = torch.as_tensor(self.diameter / 2, dtype=self.dtype)
        C = self.C
        return torch.div(C * r**2, 1 + torch.sqrt(1 - (r * C) ** 2))

    def g(self, r: Tensor) -> Tensor:
        C = self.C
        r2 = torch.pow(r, 2)
        return torch.div(C * r2, 1 + torch.sqrt(1 - r2 * torch.pow(C, 2)))

    def g_grad(self, r: Tensor) -> Tensor:
        C = self.C
        # TODO add a clamp here to avoid div by zero? or make sure fallback has an epsilon
        return torch.div(C * r, torch.sqrt(1 - torch.pow(r, 2) * torch.pow(C, 2)))

    def G(self, y: Tensor, z: Tensor) -> Tensor:
        C = self.C
        r2 = torch.pow(y, 2) + torch.pow(z, 2)
        return torch.div(C * r2, 1 + torch.sqrt(1 - r2 * torch.pow(C, 2)))

    def G_grad(self, y: Tensor, z: Tensor) -> tuple[Tensor, Tensor]:
        C = self.C
        r2 = torch.pow(y, 2) + torch.pow(z, 2)
        denom = torch.sqrt(1 - r2 * torch.pow(C, 2))
        return torch.div(y * C, denom), torch.div(z * C, denom)

    def samples2D_half(self, N: int, epsilon: float) -> Tensor:
        # If the curvature is low, use linear sampling along the y axis
        # Else, use the angular parameterization of the circle so that
        # samples are smoother, especially for high curvature circles.
        if self.C * self.diameter < 0.1:
            return sphere_samples_linear(
                curvature=self.C,
                start=0.0,
                end=self.diameter / 2 - epsilon,
                N=N,
                dtype=self.dtype,
            )
        else:
            R = 1 / self.C
            theta_max = torch.arcsin((self.diameter / 2) / torch.abs(R))
            return sphere_samples_angular(
                radius=R,
                start=0.0,
                end=(1 - epsilon) * theta_max,
                N=N,
                dtype=self.dtype,
            )

    def samples2D_full(self, N: int, epsilon: float) -> Tensor:
        # If the curvature is low, use linear sampling along the y axis
        # Else, use the angular parameterization of the circle so that
        # samples are smoother, especially for high curvature circles.
        if self.C * self.diameter < 0.1:
            return sphere_samples_linear(
                curvature=self.C,
                start=-self.diameter / 2 + epsilon,
                end=self.diameter / 2 - epsilon,
                N=N,
                dtype=self.dtype,
            )
        else:
            R = 1 / self.C
            theta_max = torch.arcsin((self.diameter / 2) / torch.abs(R))
            return sphere_samples_angular(
                radius=R,
                start=-(1 - epsilon) * theta_max,
                end=(1 - epsilon) * theta_max,
                N=N,
                dtype=self.dtype,
            )


class Parabola(SagSurface):
    "Sag surface for a parabola $X = A Y^2$"

    def __init__(
        self,
        diameter: float,
        A: int | float | nn.Parameter,
        dtype: torch.dtype = torch.float64,
    ):
        self.A = to_tensor(A, default_dtype=dtype)
        assert (
            dtype == self.A.dtype
        ), f"Inconsistent dtype between surface and parameter (surface: {dtype}) (parameter: {self.A.dtype})"

        super().__init__(diameter, dtype=dtype)

    def testname(self) -> str:
        return f"Parabola-{self.diameter:.2f}-{self.A.item():.2f}"

    def parameters(self) -> dict[str, nn.Parameter]:
        return {"A": self.A} if isinstance(self.A, nn.Parameter) else {}

    def extent_x(self) -> Tensor:
        r = torch.as_tensor(self.diameter / 2, dtype=self.dtype)
        ret = torch.as_tensor(self.A * r**2, dtype=self.dtype)
        return ret

    def g(self, r: Tensor) -> Tensor:
        return torch.mul(self.A, torch.pow(r, 2))

    def g_grad(self, r: Tensor) -> Tensor:
        return 2 * self.A * r

    def G(self, y: Tensor, z: Tensor) -> Tensor:
        return torch.mul(self.A, (y**2 + z**2))

    def G_grad(self, y: Tensor, z: Tensor) -> tuple[Tensor, Tensor]:
        return 2 * self.A * y, 2 * self.A * z

    def samples2D_half(self, N: int, epsilon: float = 1e-3) -> Tensor:
        r = torch.linspace(0, self.diameter / 2 - epsilon, N, dtype=self.dtype)
        x = self.A * r**2
        return torch.stack((x, r), dim=-1)

    def samples2D_full(self, N: int, epsilon: float = 1e-3) -> Tensor:
        r = torch.linspace(
            -self.diameter / 2 + epsilon,
            self.diameter / 2 - epsilon,
            N,
            dtype=self.dtype,
        )
        x = self.A * r**2
        return torch.stack((x, r), dim=-1)


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
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__(dtype=dtype)
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
        assert self.R.dtype == self.dtype

    def __str__(self) -> str:
        return f"SphereR({self.diameter}, {self.R.item()})"

    def testname(self) -> str:
        return str(self)

    def parameters(self) -> dict[str, nn.Parameter]:
        if isinstance(self.R, nn.Parameter):
            return {"R": self.R}
        else:
            return {}

    def radius(self) -> float:
        return self.R.item()

    def extent_x(self) -> Tensor:
        r = self.diameter / 2
        K = 1 / self.R
        return torch.div(K * r**2, 1 + torch.sqrt(1 - (r * K) ** 2))

    def bounding_radius(self) -> float:
        return math.sqrt((self.diameter / 2) ** 2 + self.extent_x() ** 2)

    def samples2D_half(self, N: int, epsilon: float) -> Tensor:
        C = 1 / self.R
        if torch.abs(C * self.diameter) < 0.1:
            # If the curvature is low, use linear sampling along the y axis
            return sphere_samples_linear(
                curvature=C,
                start=0.0,
                end=(self.diameter / 2) * (1 - epsilon),
                N=N,
                dtype=self.dtype,
            )
        else:
            # Else, use the angular parameterization of the circle so that
            # samples are smoother, especially for high curvature circles.
            theta_max = torch.arcsin((self.diameter / 2) / torch.abs(self.R))
            return sphere_samples_angular(
                radius=self.R,
                start=0.0,
                end=theta_max * (1 - epsilon),
                N=N,
                dtype=self.dtype,
            )

    def samples2D_full(self, N: int, epsilon: float) -> Tensor:
        "Like samples2D but on the entire domain"
        C = 1 / self.R
        if torch.abs(C * self.diameter) < 0.1:
            # If the curvature is low, use linear sampling along the y axis
            return sphere_samples_linear(
                curvature=C,
                start=(-self.diameter / 2) * (1 - epsilon),
                end=(self.diameter / 2) * (1 - epsilon),
                N=N,
                dtype=self.dtype,
            )
        else:
            # Else, use the angular parameterization of the circle so that
            # samples are smoother, especially for high curvature circles.
            theta_max = torch.arcsin((self.diameter / 2) / torch.abs(self.R))
            return sphere_samples_angular(
                radius=self.R,
                start=-theta_max * (1 - epsilon),
                end=theta_max * (1 - epsilon),
                N=N,
                dtype=self.dtype,
            )

    def center(self, dim: int) -> Tensor:
        if dim == 2:
            return torch.tensor([self.R, 0.0])
        else:
            return torch.tensor([self.R, 0.0, 0.0])

    def normals(self, points: Tensor) -> Tensor:
        batch, dim, dtype = points.shape[:-1], points.shape[-1], self.dtype

        # The normal is the vector from the center to the points
        center = self.center(dim)
        normals = torch.nn.functional.normalize(points - center, dim=-1)

        # We need a default value for the case where point == center, to avoid div by zero
        unit = unit_vector(dim, dtype)
        normal_at_origin = torch.tile(unit, ((*batch, 1)))

        return torch.where(
            torch.all(torch.isclose(center, points), dim=-1)
            .unsqueeze(-1)
            .expand_as(normals),
            normal_at_origin,
            normals,
        )

    def contains(self, points: Tensor, tol: Optional[float] = None) -> Tensor:

        if tol is None:
            tol = {torch.float32: 1e-3, torch.float64: 1e-7}[self.dtype]

        center = self.center(dim=points.shape[-1])
        within_outline = within_radius(self.diameter / 2 + tol, points)
        on_sphere = (
            torch.abs(
                torch.linalg.vector_norm(points - center, dim=-1) - torch.abs(self.R)
            )
            <= tol
        )
        within_extent = torch.abs(points[:, 0]) <= torch.abs(self.extent_x()) + tol

        return torch.all(
            torch.stack((within_outline, on_sphere, within_extent), dim=-1), dim=-1
        )

    def local_collide(self, P: Tensor, V: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        N, D = P.shape

        # For numerical stability, it's best if P is close to the origin
        # Bring rays origins as close as possible before solving
        init_t = init_closest_origin(self, P, V)
        P = P + init_t.unsqueeze(-1).expand_as(V) * V

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

        # TODO refactor don't rely on contains at all

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
        valid = number_of_valid_roots > 0

        return init_t + t, local_normals, valid


class Asphere(SagSurface):
    def __init__(
        self,
        diameter: float,
        R: int | float | nn.Parameter,
        K: int | float | nn.Parameter,
        A4: int | float | nn.Parameter,
        dtype: torch.dtype = torch.float64,
    ):
        self.C: torch.Tensor
        if isinstance(R, nn.Parameter):
            self.C = nn.Parameter(torch.tensor(1.0 / R.item(), dtype=R.dtype))
        else:
            self.C = torch.as_tensor(1.0 / R, dtype=dtype)
        assert self.C.dim() == 0

        self.K = to_tensor(K, default_dtype=dtype)
        self.A4 = to_tensor(A4, default_dtype=dtype)

        assert (
            dtype == self.C.dtype
        ), f"Inconsistent dtype between surface and parameter C (surface: {dtype}) (parameter: {self.C.dtype})"
        assert (
            dtype == self.K.dtype
        ), f"Inconsistent dtype between surface and parameter K (surface: {dtype}) (parameter: {self.K.dtype})"
        assert (
            dtype == self.A4.dtype
        ), f"Inconsistent dtype between surface and parameter A4 (surface: {dtype}) (parameter: {self.A4.dtype})"

        super().__init__(diameter, dtype=dtype)

    def testname(self) -> str:
        return f"Asphere-{self.diameter:.2f}-{self.C.item():.2f}-{self.K.item():.2f}-{self.A4.item():.6f}"

    def parameters(self) -> dict[str, nn.Parameter]:
        possible = {
            "C": self.C,
            "K": self.K,
            "A4": self.A4,
        }
        return {
            name: value
            for name, value in possible.items()
            if isinstance(value, nn.Parameter)
        }

    def extent_x(self) -> Tensor:
        r = torch.as_tensor(self.diameter / 2, dtype=self.dtype)
        r2 = r**2
        C, K, A4 = self.C, self.K, self.A4
        C2 = torch.pow(C, 2)
        return torch.add(
            torch.div(C * r2, 1 + torch.sqrt(1 - (1 + K) * r2 * C2)), A4 * r2**2
        )

    def samples2D_half(self, N: int, epsilon: float) -> Tensor:
        K, C, A4 = self.K, self.C, self.A4
        C2 = torch.pow(C, 2)
        dr = 1.0 - epsilon

        Y = torch.linspace(0, dr * self.diameter / 2, N, dtype=self.dtype)
        r2 = torch.pow(Y, 2)
        X = torch.div(C * r2, 1 + torch.sqrt(1 - (1 + K) * r2 * C2)) + A4 * torch.pow(
            r2, 2
        )

        return torch.stack((X, Y), dim=-1)

    def samples2D_full(self, N: int, epsilon: float) -> Tensor:
        K, C, A4 = self.K, self.C, self.A4
        C2 = torch.pow(C, 2)
        dr = 1.0 - epsilon

        Y = torch.linspace(
            -dr * self.diameter / 2,
            dr * self.diameter / 2,
            N,
            dtype=self.dtype,
        )
        r2 = torch.pow(Y, 2)
        X = torch.div(C * r2, 1 + torch.sqrt(1 - (1 + K) * r2 * C2)) + A4 * torch.pow(
            r2, 2
        )

        return torch.stack((X, Y), dim=-1)

    def g(self, r: Tensor) -> Tensor:
        r2 = torch.pow(r, 2)
        K, C, A4 = self.K, self.C, self.A4
        C2 = torch.pow(C, 2)
        return torch.div(
            C * r2, 1 + torch.sqrt(1 - (1 + K) * r2 * C2)
        ) + A4 * torch.pow(r2, 2)

    def g_grad(self, r: Tensor) -> Tensor:
        r2 = torch.pow(r, 2)
        K, C, A4 = self.K, self.C, self.A4
        C2 = torch.pow(C, 2)

        return torch.div(C * r, torch.sqrt(1 - (1 + K) * r2 * C2)) + 4 * A4 * torch.pow(
            r, 3
        )

    def G(self, y: Tensor, z: Tensor) -> Tensor:
        K, C, A4 = self.K, self.C, self.A4
        C2 = torch.pow(C, 2)
        r2 = y**2 + z**2

        return torch.div(
            C * r2, 1 + torch.sqrt(1 - (1 + K) * r2 * C2)
        ) + A4 * torch.pow(r2, 2)

    def G_grad(self, y: Tensor, z: Tensor) -> tuple[Tensor, Tensor]:
        K, C, A4 = self.K, self.C, self.A4
        C2 = torch.pow(C, 2)
        r2 = y**2 + z**2

        denom = torch.sqrt(1 - (1 + K) * r2 * C2)
        coeffs_term = 4 * A4 * r2

        return (C * y) / denom + y * coeffs_term, (C * z) / denom + z * coeffs_term
