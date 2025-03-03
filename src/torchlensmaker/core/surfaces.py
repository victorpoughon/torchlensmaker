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
from torchlensmaker.core.geometry import unit_vector, within_radius

from torch.linalg import vector_norm

from typing import Any

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
        super().__init__(dtype)
        self.outline = outline

    def parameters(self) -> dict[str, nn.Parameter]:
        return {}

    def samples2D_half(self, N: int, epsilon: float = 1e-3) -> Tensor:
        r = torch.linspace(0, self.outline.max_radius(), N, dtype=self.dtype)
        return torch.stack((torch.zeros(N), r), dim=-1)

    def samples2D_full(self, N: int, epsilon: float = 1e-3) -> Tensor:
        maxr = self.outline.max_radius()
        r = torch.linspace(-maxr, maxr, N, dtype=self.dtype)
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
        batch, dim = points.shape[:-1], points.shape[-1]
        normal = -unit_vector(dim=dim, dtype=self.dtype)
        return torch.tile(normal, (*batch, 1))

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

        return torch.abs(F(points)) < tol

    def local_collide(self, P: Tensor, V: Tensor) -> tuple[Tensor, Tensor, Tensor]:

        t = self.collision_method(self, P, V)

        local_points = P + t.unsqueeze(1).expand_as(V) * V
        local_normals = self.normals(local_points)

        # If there is no intersection, collision detection won't converge
        # and points will not be on the surface
        # So verify intersection here and filter points
        # that aren't on the surface
        # TODO better tolerance configuration based on sampling dtype
        valid = self.contains(local_points, tol=1e-4)

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


class Parabola(ImplicitSurface):
    def __init__(
        self,
        diameter: float,
        a: int | float | nn.Parameter,
        **kwargs,
    ):
        super().__init__(**kwargs)
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
        r = torch.linspace(0, self.diameter/2, N)
        x = self.a * r**2
        return torch.stack((x, r), dim=-1)
    
    def samples2D_full(self, N: int, epsilon: float = 1e-3) -> Tensor:
        r = torch.linspace(-self.diameter/2, self.diameter/2, N)
        x = self.a * r**2
        return torch.stack((x, r), dim=-1)

    def extent_x(self) -> Tensor:
        r = torch.as_tensor(self.diameter/2, dtype=self.dtype)
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


class DiameterBandSurface(ImplicitSurface):
    def __init__(self, Ax, Ar, **kwargs):
        super().__init__(**kwargs)
        self.Ax = Ax
        self.Ar = Ar

    def f(self, points: Tensor) -> Tensor:
        assert points.shape[-1] == 2
        X, R = points.unbind(-1)
        Ax, Ar = self.Ax, self.Ar
        return (X - Ax) ** 2 + (torch.abs(R) - Ar) ** 2

    def f_grad(self, points: Tensor) -> Tensor:
        assert points.shape[-1] == 2
        X, R = points.unbind(-1)
        Ax, Ar = self.Ax, self.Ar
        return torch.stack(
            (2 * (X - Ax), torch.sign(R) * 2 * (torch.abs(R) - Ar)), dim=-1
        )

    def F(self, points: Tensor) -> Tensor:
        assert points.shape[-1] == 3
        X, Y, Z = points.unbind(-1)
        R2 = Y**2 + Z**2
        Ax, Ar = self.Ax, self.Ar
        return (X - Ax) ** 2 + (torch.sqrt(R2) - Ar) ** 2

    def F_grad(self, points: Tensor) -> Tensor:
        assert points.shape[-1] == 3
        X, Y, Z = points.unbind(-1)
        R2 = Y**2 + Z**2
        Ax, Ar = self.Ax, self.Ar
        quot = Ar / torch.sqrt(R2)
        return torch.stack((2 * (X - Ax), 2 * Y - quot, 2 * Z - quot), dim=-1)


class DiameterBandSurfaceSq(ImplicitSurface):
    "Square root version of diameter band surface"

    def __init__(self, Ax, Ar, **kwargs):
        super().__init__(**kwargs)
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
    Implicit surface defined by a sag function.

    A sag function g(r) is a one dimensional real valued function that describes
    a surface x coordinate in an arbitrary meridional plane (x,r) as a function
    of the distance to the principal axis: x = g(r).

    Derived classes provide the sag function and its gradient in
    both 2 and 3 dimensions. This class then uses it to create the implicit
    function F representing the corresponding implicit surface.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
        return self.g(r) - x

    def f_grad(self, points: Tensor) -> Tensor:
        assert points.shape[-1] == 2
        x, r = points.unbind(-1)
        return torch.stack((-torch.ones_like(x), self.g_grad(r)), dim=-1)
    
    def F(self, points: Tensor) -> Tensor:
        assert points.shape[-1] == 3
        x, y, z = points.unbind(-1)
        return self.G(y, z) - x

    def F_grad(self, points: Tensor) -> Tensor:
        assert points.shape[-1] == 3
        x, y, z = points.unbind(-1)
        grad_y, grad_z = self.G_grad(y, z)
        return torch.stack((-torch.ones_like(x), grad_y, grad_z), dim=-1)


class CompositeImplicitSurface(ImplicitSurface):
    """
    Composes two implicit surfaces spatially with a mask function.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.inner_surface = None # To be set by derived class after initialization
        self.fallback_surface_type = None # To be set by derived class after initialization
    
    def mask_function(self, surface: ImplicitSurface, points: Tensor) -> Tensor:
        raise NotImplementedError
    
    def fallback_surface(self) -> ImplicitSurface:
        # TODO dtype
        return self.fallback_surface_type(Ax=self.inner_surface.extent_x(), Ar=self.inner_surface.diameter/2, dtype=self.dtype)
    
    def f(self, points: Tensor) -> Tensor:
        assert points.shape[-1] == 2
        mask = self.mask_function(self.inner_surface, points)
        fallback = self.fallback_surface()
        return torch.where(mask, self.inner_surface.f(points), fallback.f(points))

    def f_grad(self, points: Tensor) -> Tensor:
        assert points.shape[-1] == 2
        mask = self.mask_function(self.inner_surface, points)
        fallback = self.fallback_surface()
        return torch.where(mask.unsqueeze(-1).expand(*mask.size(), 2), self.inner_surface.f_grad(points), fallback.f_grad(points))
    
    def F(self, points: Tensor) -> Tensor:
        assert points.shape[-1] == 3
        mask = self.mask_function(self.inner_surface, points)
        fallback = self.fallback_surface()
        return torch.where(mask, self.inner_surface.F(points), fallback.F(points))

    def F_grad(self, points: Tensor) -> Tensor:
        assert points.shape[-1] == 3
        mask = self.mask_function(self.inner_surface, points)
        fallback = self.fallback_surface()
        return torch.where(mask.unsqueeze(-1).expand(*mask.size(), 3), self.inner_surface.F_grad(points), fallback.F_grad(points))


class SphereSag(SagSurface):
    "Sag surface for Sphere parameterized with curvature"

    def __init__(self, diameter: Tensor, C: Tensor, **kwargs):
        super().__init__(**kwargs)
        self.diameter = diameter
        self.C = C
    
    def parameters(self) -> dict[str, nn.Parameter]:
        if (isinstance(self.C, nn.Parameter)):
            return {"C": self.C}
        else:
            return {}

    def extent_x(self) -> Tensor:
        r = torch.as_tensor(self.diameter/2, dtype=self.dtype)
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
        denom = torch.sqrt(1 - r2*torch.pow(C, 2))
        return torch.div(y * C, denom), torch.div(z * C, denom)

    def samples2D_half(self, N: int, epsilon: float) -> Tensor:
        # If the curvature is low, use linear sampling along the y axis
        # Else, use the angular parameterization of the circle so that
        # samples are smoother, especially for high curvature circles.
        if self.C * self.diameter < 0.1:
            return sphere_samples_linear(
                curvature=self.C,
                start=0.0,
                end=self.diameter/2 - epsilon,
                N=N,
                dtype=self.dtype,
            )
        else:
            R = 1 / self.C
            theta_max = torch.arcsin((self.diameter/2) / torch.abs(R))
            return sphere_samples_angular(
                radius=R, start=0.0, end=theta_max - epsilon, N=N,
                dtype=self.dtype
            )

    def samples2D_full(self, N: int, epsilon: float) -> Tensor:
        # If the curvature is low, use linear sampling along the y axis
        # Else, use the angular parameterization of the circle so that
        # samples are smoother, especially for high curvature circles.
        if self.C * self.diameter < 0.1:
            return sphere_samples_linear(
                curvature=self.C,
                start=-self.diameter/2 + epsilon,
                end=self.diameter/2 - epsilon,
                N=N,
                dtype=self.dtype,
            )
        else:
            R = 1 / self.C
            theta_max = torch.arcsin((self.diameter/2) / torch.abs(R))
            return sphere_samples_angular(
                radius=R, start=-theta_max + epsilon, end=theta_max - epsilon, N=N, dtype=self.dtype
            )


class Sphere(CompositeImplicitSurface):
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
        fallback_surface_type: Any = DiameterBandSurfaceSq,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.diameter = diameter

        if (R is not None and C is not None) or (R is None and C is None):
            raise RuntimeError(
                "Sphere must be initialized with exactly one of R (radius) or C (curvature)."
            )

        self.C: torch.Tensor
        if C is None:
            if torch.abs(torch.as_tensor(R)) <= diameter / 2:
                raise RuntimeError(
                    f"Sphere radius (R={R}) must be strictly greater than half the surface diameter (D/2={diameter/2}) "
                    f"(To model an exact half-sphere, use SphereR)."
                )

            if isinstance(R, nn.Parameter):
                C_tensor = nn.Parameter(torch.tensor(1.0 / R.item(), dtype=self.dtype))
            else:
                C_tensor = torch.as_tensor(1.0 / R, dtype=self.dtype)
        else:
            if isinstance(C, nn.Parameter):
                C_tensor = C
            else:
                C_tensor = torch.as_tensor(C, dtype=self.dtype)

        assert C_tensor.dim() == 0
        assert C_tensor.dtype == self.dtype
        
        self.inner_surface = SphereSag(diameter=diameter, C=C_tensor, **kwargs)
        self.fallback_surface_type = fallback_surface_type
    
    def mask_function(self, surface: ImplicitSurface, points: Tensor) -> Tensor:
        # TODO bbox / domain
        return within_radius(self.inner_surface.diameter / 2, points)

    def radius(self) -> float:
        "Utility function to get radius from internal curvature"
        return 1/self.inner_surface.C.item()

    def testname(self) -> str:
        return f"Sphere-{self.inner_surface.diameter:.2f}-{self.inner_surface.C.item():.2f}"
    
    def extent_x(self) -> Tensor:
        return self.inner_surface.extent_x()

    def parameters(self) -> dict[str, nn.Parameter]:
        return self.inner_surface.parameters()
    
    def samples2D_half(self, N: int, epsilon: float) -> Tensor:
        return self.inner_surface.samples2D_half(N, epsilon)

    def samples2D_full(self, N: int, epsilon: float) -> Tensor:
        return self.inner_surface.samples2D_full(N, epsilon)


class SphereC(Sphere):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fallback_surface_type = DiameterBandSurface
    
    def testname(self) -> str:
        return "SphereC"


# for SphereSag make sure there is still an inner where() for safe backwards when masking nans
# just set all outside domain to zero basically, will be replaced by composite surface anyway

# think if we need some tolerance / epsilon on mask boundary

# Sphere = CompositeSurface(within_radius, SagSphere, DiameterBandSurface)

# add domain to surfaces?
# clean up bbox / domain of surface instead of extent_x / diameter/2

class SphereOld(ImplicitSurface):

    def __init__(
        self,
        diameter: float,
        R: int | float | nn.Parameter | None = None,
        C: int | float | nn.Parameter | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
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
        assert self.C.dtype == self.dtype

    def testname(self) -> str:
        return f"SphereOld-{self.diameter:.2f}-{self.C.item():.2f}"

    def parameters(self) -> dict[str, nn.Parameter]:
        if isinstance(self.C, nn.Parameter):
            return {"C": self.C}
        else:
            return {}

    def radius(self) -> float:
        return 1 / self.C.item()

    def extent_x(self) -> Tensor:
        r = torch.as_tensor(self.diameter/2, dtype=self.dtype)
        C = self.C
        return torch.div(C * r**2, 1 + torch.sqrt(1 - (r * C) ** 2))

    def samples2D_half(self, N: int, epsilon: float) -> Tensor:
        # If the curvature is low, use linear sampling along the y axis
        # Else, use the angular parameterization of the circle so that
        # samples are smoother, especially for high curvature circles.
        if self.C * self.diameter < 0.1:
            return sphere_samples_linear(
                curvature=self.C,
                start=0.0,
                end=self.diameter/2 - epsilon,
                N=N,
                dtype=self.dtype,
            )
        else:
            R = 1 / self.C
            theta_max = torch.arcsin((self.diameter/2) / torch.abs(R))
            return sphere_samples_angular(
                radius=R, start=0.0, end=theta_max - epsilon, N=N,
                dtype=self.dtype,
            )

    def samples2D_full(self, N: int, epsilon: float) -> Tensor:
        # If the curvature is low, use linear sampling along the y axis
        # Else, use the angular parameterization of the circle so that
        # samples are smoother, especially for high curvature circles.
        if self.C * self.diameter < 0.1:
            return sphere_samples_linear(
                curvature=self.C,
                start=-self.diameter/2 + epsilon,
                end=self.diameter/2 - epsilon,
                N=N,
                dtype=self.dtype,
            )
        else:
            R = 1 / self.C
            theta_max = torch.arcsin((self.diameter/2) / torch.abs(R))
            return sphere_samples_angular(
                radius=R, start=-theta_max + epsilon, end=theta_max - epsilon, N=N, dtype=self.dtype
            )

    def edge_points(self) -> tuple[Tensor, Tensor]:
        "2D edge points"

        A = self.extent(dim=2) + torch.tensor([0.0, self.diameter / 2])
        B = self.extent(dim=2) - torch.tensor([0.0, self.diameter / 2])
        return A, B


    def f(self, points: Tensor) -> Tensor:
        assert points.shape[-1] == 2
        X, R = points.select(-1, 0), points.select(-1, 1)
        R2 = torch.pow(R, 2)
        C = self.C

        # For points beyond the diameter, use distance to the edge point
        A, B = self.edge_points()
        top_fallback = vector_norm(points - A, dim=-1)
        bottom_fallback = vector_norm(points - B, dim=-1)

        # Implicit function based on sphere sag function
        radicand = 1 - R2 * C**2
        safe_radicand = torch.clamp(radicand, min=0.0)
        circle = torch.div(C * R2, 1 + torch.sqrt(safe_radicand)) - X

        assert circle.shape == top_fallback.shape == bottom_fallback.shape

        return torch.where(
            within_radius(self.diameter/2, points),
            circle,
            torch.where(R > 0, top_fallback, bottom_fallback),
        )

    def f_grad(self, points: Tensor) -> Tensor:
        assert points.shape[-1] == 2
        X, R = points.select(-1, 0), points.select(-1, 1)
        R2 = torch.pow(R, 2)
        C = self.C

        # For points beyond the diameter, use distance to the edge point
        A, B = self.edge_points()
        normA, normB = vector_norm(points - A, dim=-1), vector_norm(points - B, dim=-1)

        # Derivative of distance
        top_fallback = torch.stack(((X - A[0]) / normA, (R - A[1]) / normA), dim=-1)
        bottom_fallback = torch.stack(((X - B[0]) / normB, (R - B[1]) / normB), dim=-1)

        radicand = 1 - R2 * C**2

        # clamp to a non zero epsilon to avoid both sqrt(<0) and div by zero
        safe_radicand = torch.clamp(radicand, min=1e-4)
        grady = torch.div((C * R), torch.sqrt(safe_radicand))

        circle = torch.stack((-torch.ones_like(X), grady), dim=-1)

        assert circle.shape == top_fallback.shape == bottom_fallback.shape

        return torch.where(
            within_radius(self.diameter/2, points).unsqueeze(-1).expand(-1, 2),
            circle,
            torch.where(
                R.unsqueeze(-1).expand(-1, 2) > 0, top_fallback, bottom_fallback
            ),
        )

    # TODO zone mask in 3D
    def F(self, points: Tensor) -> Tensor:
        assert points.shape[-1] == 3
        X, Y, Z = points.select(-1, 0), points.select(-1, 1),  points.select(-1, 2)
        C = self.C
        R2 = Y**2 + Z**2

        return torch.div(C * R2, 1 + torch.sqrt(1 - R2 * C**2)) - X

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
        super().__init__(**kwargs)
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
        r = self.diameter/2
        K = 1 / self.R
        return torch.div(K * r**2, 1 + torch.sqrt(1 - (r * K) ** 2))

    def samples2D_half(self, N: int, epsilon: float) -> Tensor:
        C = 1 / self.R
        if torch.abs(C * self.diameter) < 0.1:
            # If the curvature is low, use linear sampling along the y axis
            return sphere_samples_linear(
                curvature=C, start=0.0, end=self.diameter/2 - epsilon, N=N, dtype=self.dtype
            )
        else:
            # Else, use the angular parameterization of the circle so that
            # samples are smoother, especially for high curvature circles.
            theta_max = torch.arcsin((self.diameter/2) / torch.abs(self.R))
            return sphere_samples_angular(radius=self.R, start=0.0, end=theta_max - epsilon, N=N, dtype=self.dtype)

    def samples2D_full(self, N: int, epsilon: float) -> Tensor:
        "Like samples2D but on the entire domain"
        C = 1 / self.R
        if torch.abs(C * self.diameter) < 0.1:
            # If the curvature is low, use linear sampling along the y axis
            return sphere_samples_linear(
                curvature=C,
                start=-self.diameter/2 + epsilon,
                end=self.diameter/2 - epsilon,
                N=N,
                dtype=self.dtype
            )
        else:
            # Else, use the angular parameterization of the circle so that
            # samples are smoother, especially for high curvature circles.
            theta_max = torch.arcsin((self.diameter/2) / torch.abs(self.R))
            return sphere_samples_angular(
                radius=self.R, start=-theta_max + epsilon, end=theta_max - epsilon, N=N, dtype=self.dtype
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
            torch.all(torch.isclose(center, points), dim=-1).unsqueeze(-1).expand_as(normals),
            normal_at_origin,
            normals,
        )

    def contains(self, points: Tensor, tol: float = 1e-6) -> Tensor:
        center = self.center(dim=points.shape[-1])
        within_outline = within_radius(self.diameter/2 + tol, points)
        on_sphere = torch.abs(torch.linalg.vector_norm(points - center, dim=-1) - torch.abs(self.R)) <= tol
        within_extent = torch.abs(points[:, 0]) <= torch.abs(self.extent_x()) + tol

        return torch.all(torch.stack((within_outline, on_sphere, within_extent), dim=-1), dim=-1)

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
        super().__init__(**kwargs)
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
        r2 = (self.diameter/2) ** 2
        C, K, A4 = self.C, self.K, self.A4
        C2 = torch.pow(C, 2)
        return torch.div(C * r2, 1 + torch.sqrt(1 - (1 + K) * r2 * C2)) + A4 * r2**2

    def samples2D_half(self, N: int, epsilon: float) -> Tensor:
        K, C, A4 = self.K, self.C, self.A4
        C2 = torch.pow(C, 2)

        Y = torch.linspace(0, self.diameter/2 - epsilon, N)
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
