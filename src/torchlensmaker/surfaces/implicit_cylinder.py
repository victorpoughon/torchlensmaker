import torch
import torch.func
from torchlensmaker.surfaces.implicit_surface import ImplicitSurface


def sd_capped_cylinder_x(
    p: torch.Tensor, xmin: torch.Tensor, xmax: torch.Tensor, tau: torch.Tensor
) -> torch.Tensor:
    """Signed distance function for an X-axis aligned capped cylinder.

    Args:
        p: Input points (..., 3)
        xmin: Start X-coordinate (scalar tensor)
        xmax: End X-coordinate (scalar tensor)
        tau: Cylinder radius (scalar tensor)

    Returns:
        Signed distances (...,)
    """
    # Radial distance in YZ-plane
    radial = torch.norm(p[..., 1:3], dim=-1) - tau

    # Axial distance along X-axis
    axial = torch.maximum(xmin - p[..., 0], p[..., 0] - xmax)

    # Combine distance components
    d = torch.stack([radial, axial], dim=-1)

    # Calculate SDF components
    max_d = torch.max(d, dim=-1)[0]
    min_term = torch.minimum(max_d, torch.tensor(0.0, device=p.device))
    clamped_d = torch.clamp(d, min=0.0)
    length_term = torch.norm(clamped_d, dim=-1)

    return torch.add(min_term, length_term)

def sqrt_safe2(v):
    return torch.sqrt(torch.clamp(v, min=torch.finfo(v.dtype).tiny))

def sccx_unbatched(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    xmin: torch.Tensor,
    xmax: torch.Tensor,
    tau: torch.Tensor,
) -> torch.Tensor:
    "Unbatched version of sd_capped_cylinder_x"

    # Radial distance in YZ-plane
    radial = sqrt_safe2(y**2 + z**2) - tau

    # Axial distance along X-axis
    axial = torch.maximum(xmin - x, x - xmax)

    # Combine distance components
    d = torch.stack([radial, axial], dim=-1)

    # Calculate SDF components
    max_d = torch.max(d, dim=-1)[0]
    min_term = torch.minimum(max_d, torch.tensor(0.0, device=x.device))
    clamped_d = torch.clamp(d, min=0.0)
    length_term = torch.linalg.vector_norm(clamped_d, dim=-1)

    return torch.add(min_term, length_term)


class ImplicitCylinder(ImplicitSurface):
    "Cylinder aligned with the X axis"

    def __init__(self, xmin: torch.Tensor, xmax: torch.Tensor, tau: torch.Tensor):
        self.xmin = xmin
        self.xmax = xmax
        self.tau = tau
        super().__init__()

    def F(self, points: torch.Tensor) -> torch.Tensor:
        return sd_capped_cylinder_x(points, self.xmin, self.xmax, self.tau)

    def F_grad(self, points: torch.Tensor) -> torch.Tensor:
        def bind(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
            return sccx_unbatched(x, y, z, self.xmin, self.xmax, self.tau)

        # Add batch dimensions
        G = torch.func.grad(bind, (0, 1, 2))
        for i in range(points.dim() - 1):
            G = torch.vmap(G)

        x, y, z = points.unbind(-1)
        dx, dy, dz = G(x, y, z)
        return torch.stack((dx, dy, dz), dim=-1)
