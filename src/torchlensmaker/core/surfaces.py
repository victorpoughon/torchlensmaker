# This file is part of Torch Lens Maker
# Copyright (C) 2025 Victor Poughon
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

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
from torchlensmaker.core.cylinder_collision import rays_cylinder_collision, rays_rectangle_collision

from torchlensmaker.core.sag_functions import (
    SagFunction,
    Spherical,
    Parabolic,
    Aspheric,
    Conical,
    SagSum,
)

from typing import Optional, Any

# shorter for type annotations
Tensor = torch.Tensor


class LocalSurface:
    """
    Base class for surfaces
    """

    def __init__(self, dtype: torch.dtype = torch.float64):
        self.dtype = dtype

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
        Input points are expected to be on, or at least near, the surface
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

    def to_dict(self, dim: int) -> dict[str, Any]:
        """
        Convert to a dictionary for JSON serialization
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
            init_closest_origin(self, P, V),  # Default for vertical rays
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

    def to_dict(self, _dim: int) -> dict[str, Any]:
        return {
            "type": "surface-plane",
            "radius": self.outline.max_radius(),
            "clip_planes": self.outline.clip_planes(),
        }


class SquarePlane(Plane):
    def __init__(self, side_length: float, dtype: torch.dtype = torch.float64):
        self.side_length = side_length
        super().__init__(SquareOutline(side_length), dtype)


class CircularPlane(Plane):
    "aka disk"

    def __init__(self, diameter: float, dtype: torch.dtype = torch.float64):
        self.diameter = diameter
        super().__init__(CircularOutline(diameter), dtype)


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
    
    def bcyl(self) -> Tensor:
        raise NotImplementedError

    def rmse(self, points: Tensor) -> float:
        N = sum(points.shape[:-1])
        return torch.sqrt(torch.sum(self.Fd(points) ** 2) / N).item()

    def local_collide(self, P: Tensor, V: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        assert P.dtype == V.dtype == self.dtype
        assert P.shape == V.shape
        
        N, dim = P.shape
        dtype = P.dtype

        with torch.no_grad():
            # Get bounding cylinder and compute ray-cylinder intersection
            xmin, xmax, tau = self.bcyl().unbind()
            if dim == 3:
                t1, t2, hit_mask = rays_cylinder_collision(P, V, xmin, xmax, tau)
            else:
                t1, t2, hit_mask = rays_rectangle_collision(P, V, xmin, xmax, -tau, tau)

            # Split rays into definitly not colliding, and possibly colliding ("maybe")
            P_maybe, V_maybe = P[hit_mask], V[hit_mask]
            tmin, tmax = t1[hit_mask], t2[hit_mask]

        # Run iterative collision method on possibly colliding rays
        # TODO if no "maybe rays", skip this call, because tmin tmax are empty when N=0
        t = self.collision_method(self, P_maybe, V_maybe, tmin, tmax, history=False).t

        # two kinds of non colliding rays at this points:
        # - non colliding from before bounding cylinder check
        # - non colliding after iterations complete, t will make a non colliding point

        local_points = P_maybe + t.unsqueeze(-1).expand_as(V_maybe) * V_maybe
        local_normals = self.normals(local_points)
        valid = self.contains(local_points)

        # final t: t made into total N shape
        hit_mask_indices = hit_mask.nonzero().squeeze(-1)
        final_t = torch.zeros((N,), dtype=t.dtype).index_put(
            (hit_mask_indices,), t
        )

        default_normal = unit_vector(dim, dtype)
        final_normals = (
            default_normal.unsqueeze(0)
            .expand(N, dim)
            .index_put((hit_mask_indices,), local_normals)
        )

        # final mask: cylinder_hit_mask combined with contains(local_points) mask
        final_valid = torch.full((N,), False).index_put(
            (hit_mask_indices,), valid
        )

        return final_t, final_normals, final_valid

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


class SagSurface(ImplicitSurface):
    """
    Axially symmetric implicit surface defined by a sag function.

    A sag function g(r) is a one dimensional real valued function that describes
    a surface x coordinate in an arbitrary meridional plane (x,r) as a function
    of the distance to the principal axis: x = g(r).

    Sag function classes provide the sag function and its gradient in
    both 2 and 3 dimensions. This class then uses it to create the implicit
    function F representing the corresponding implicit surface.

    The sag function is assumed defined only on the domain (- diameter/2 ;
    diameter / 2) in the meridional plane. Outside of this domain, a fallback
    function is used (implemented by DiameterBandSurfaceSq).
    """

    def __init__(
        self,
        diameter: float,
        sag_function: SagFunction,
        collision_method: CollisionMethod = default_collision_method,
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__(collision_method=collision_method, dtype=dtype)
        self.diameter = diameter
        self.sag_function = sag_function

    def mask_function(self, points: Tensor) -> Tensor:
        return within_radius(self.diameter / 2, points)

    def parameters(self) -> dict[str, nn.Parameter]:
        return self.sag_function.parameters()

    # TODO remove?
    def bounding_radius(self) -> float:
        """
        Any point on the surface has a distance to the center that is less
        than (or equal) to the bounding radius
        """
        return math.sqrt((self.diameter / 2) ** 2 + self.extent_x() ** 2)

    def tau(self) -> Tensor:
        "Half-diameter and normalization factor"
        return torch.as_tensor(self.diameter / 2, dtype=self.dtype)

    def f(self, points: Tensor) -> Tensor:
        "points are assumed to be within the bcyl domain"
        assert points.shape[-1] == 2
        x, r = points.unbind(-1)
        return self.sag_function.g(r, self.tau()) - x

    def f_grad(self, points: Tensor) -> Tensor:
        assert points.shape[-1] == 2
        x, r = points.unbind(-1)
        return torch.stack(
            (-torch.ones_like(x), self.sag_function.g_grad(r, self.tau())), dim=-1
        )

    def F(self, points: Tensor) -> Tensor:
        assert points.shape[-1] == 3
        x, y, z = points.unbind(-1)
        return self.sag_function.G(y, z, self.tau()) - x

    def F_grad(self, points: Tensor) -> Tensor:
        assert points.shape[-1] == 3
        x, y, z = points.unbind(-1)
        grad_y, grad_z = self.sag_function.G_grad(y, z, self.tau())
        return torch.stack((-torch.ones_like(x), grad_y, grad_z), dim=-1)

    def extent_x(self) -> Tensor:
        return torch.max(torch.abs(self.sag_function.bounds(self.tau())))

    def bcyl(self) -> Tensor:
        """Bounding cylinder
        Returns a tensor of shape (3,) where entries are [xmin, xmax, radius]
        """

        tau = torch.tensor(self.diameter / 2, dtype=self.dtype)
        return torch.cat(
            (
                self.sag_function.bounds(tau),
                tau.unsqueeze(0),
            ),
            dim=0,
        )

    def contains(self, points: Tensor, tol: Optional[float] = None) -> Tensor:
        if tol is None:
            tol = {torch.float32: 1e-4, torch.float64: 1e-7}[self.dtype]

        N, dim = points.shape

        # Check points are within the diameter
        r = torch.abs(points[:, 1]) if dim == 2 else torch.sqrt(points[:, 1] ** 2 + points[:, 2] ** 2)
        within_diameter = r <= self.diameter

        tau = self.tau()
        zeros1d = torch.zeros_like(points[:, 1])
        zeros2d = torch.zeros_like(r)

        # If within diameter, check the sag equation x = g(r)
        if dim == 2:
            safe_input = torch.where(within_diameter, r, zeros2d)
            sagG = self.sag_function.g(safe_input, tau)
            G = torch.where(within_diameter, sagG, zeros2d)
        else:
            safe_input_y = torch.where(within_diameter, points[:, 1], zeros1d)
            safe_input_z = torch.where(within_diameter, points[:, 2], zeros1d)
            sagG = self.sag_function.G(safe_input_y, safe_input_z, tau)
            G = torch.where(within_diameter, sagG, zeros2d)

        within_tol = torch.abs(G - points[:, 0]) < tol
        return torch.logical_and(within_diameter, within_tol)

    def samples2D_full(self, N, epsilon):
        start = -(1 - epsilon) * self.diameter / 2
        end = (1 - epsilon) * self.diameter / 2
        r = torch.linspace(start, end, N, dtype=self.dtype)
        x = self.sag_function.g(r, self.tau())
        return torch.stack((x, r), dim=-1)

    def samples2D_half(self, N, epsilon):
        start = 0.0
        end = (1 - epsilon) * self.diameter / 2
        r = torch.linspace(start, end, N, dtype=self.dtype)
        x = self.sag_function.g(r, self.tau())
        return torch.stack((x, r), dim=-1)

    def to_dict(self, dim: int) -> dict[str, Any]:
        return {
            "type": "surface-sag",
            "diameter": self.diameter,
            "sag-function": self.sag_function.to_dict(dim),
            "bcyl": self.bcyl().tolist(),
        }


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
        normalize: bool = False,
        collision_method: CollisionMethod = default_collision_method,
        dtype: torch.dtype = torch.float64,
    ):
        if (R is not None and C is not None) or (R is None and C is None):
            raise RuntimeError(
                "Sphere must be initialized with exactly one of R (radius) or C (curvature)."
            )

        C_tensor: torch.Tensor | nn.Parameter
        if C is None and R is not None:
            if isinstance(R, nn.Parameter):
                C_tensor = nn.Parameter(torch.tensor(1.0 / R.item(), dtype=R.dtype))
            else:
                C_tensor = torch.as_tensor(1.0 / R, dtype=dtype)
        else:
            if isinstance(C, nn.Parameter):
                C_tensor = C
            else:
                C_tensor = torch.as_tensor(C, dtype=dtype)

        # Domain error check
        tau = diameter / 2
        C_unnormed = C_tensor / tau if normalize else C_tensor
        if torch.abs(1.0 / C_unnormed) <= tau:
            raise RuntimeError(
                f"Sphere radius must be strictly greater than half the surface diameter (D/2={diameter / 2}) "
                f"(To model an exact half-sphere, use SphereR)."
            )

        assert C_tensor.dim() == 0
        assert C_tensor.dtype == dtype

        sag_function = Spherical(C_tensor, normalize)

        super().__init__(diameter, sag_function, collision_method, dtype)

    def radius(self) -> float:
        "Utility function to get radius from internal curvature"
        return torch.div(
            torch.tensor(1.0, dtype=self.dtype), self.sag_function.C
        ).item()


class Parabola(SagSurface):
    "Sag surface for a parabola $X = A R^2$"

    def __init__(
        self,
        diameter: float,
        A: int | float | nn.Parameter,
        normalize: bool = False,
        dtype: torch.dtype = torch.float64,
    ):
        if isinstance(A, torch.Tensor):
            assert A.dtype == dtype
        A_tensor = to_tensor(A, default_dtype=dtype)
        sag_function = Parabolic(A_tensor, normalize)
        super().__init__(diameter, sag_function, dtype=dtype)
    
    @property
    def A(self):
        return self.sag_function.A


# TODO
class Conic(SagSurface): ...


class Asphere(SagSurface):
    """
    Sag surface for the Asphere model

    An asphere is a conical base + aspheric coefficients
    """

    def __init__(
        self,
        diameter: float,
        R: int | float | nn.Parameter,
        K: int | float | nn.Parameter,
        coefficients: list[int | float] | torch.Tensor,
        normalize_conical: bool = False,
        normalize_aspheric: bool = False,
        collision_method: CollisionMethod = default_collision_method,
        dtype: torch.dtype = torch.float64,
    ):
        if isinstance(R, torch.Tensor):
            assert R.dtype == dtype
            assert R.dim() == 0
        if isinstance(K, torch.Tensor):
            assert K.dtype == dtype
            assert K.dim() == 0
        if isinstance(coefficients, torch.Tensor):
            assert coefficients.dtype == dtype
            assert coefficients.dim() == 1

        C_tensor = to_tensor(1.0 / R, default_dtype=dtype)
        K_tensor = to_tensor(K, default_dtype=dtype)
        coefficients_tensor = to_tensor(coefficients, default_dtype=dtype)

        # This prevents against initializing with out of domain K value,
        # but not getting there during optimization.
        # TODO add an Asphere model reparameterized with softplus
        tau = diameter / 2
        C_unnormed = C_tensor / tau if normalize_conical else C_tensor
        if diameter / 2 >= torch.sqrt(1 / (C_unnormed**2 * (1 + K_tensor))):
            raise ValueError(f"Out of domain asphere parameters {C_tensor} {K_tensor}")

        sag_function = SagSum(
            [
                Conical(C_tensor, K_tensor, normalize_conical),
                Aspheric(coefficients_tensor, normalize_aspheric),
            ]
        )

        super().__init__(diameter, sag_function, collision_method, dtype)


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

    def to_dict(self, _dim: int) -> dict[str, Any]:
        return {
            "type": "surface-sphere-r",
            "diameter": self.diameter,
            "R": self.R.item(),
        }
