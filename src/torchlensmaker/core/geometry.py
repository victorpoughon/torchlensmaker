import torch

from torchlensmaker.core.rot2d import rot2d
from torchlensmaker.core.rot3d import euler_angles_to_matrix

Tensor = torch.Tensor


def unit_vector(dim: int, dtype: torch.dtype) -> Tensor:
    "Unit vector along the X axis"
    return torch.cat((torch.ones(1, dtype=dtype), torch.zeros(dim - 1, dtype=dtype)))

def rotated_unit_vector(angles: Tensor, dim: int) -> Tensor:
    """
    Rotated unit X vector in 2D or 3D
    angles is batched with shape (N, 2|3)
    """

    dtype = angles.dtype
    N = angles.shape[0]
    if dim == 2:
        unit = torch.tensor([1.0, 0.0], dtype=dtype)
        return rot2d(unit, angles)
    else:
        unit = torch.tensor([1.0, 0.0, 0.0], dtype=dtype)
        thetas = torch.column_stack(
            (
                torch.zeros(N, dtype=dtype),
                angles,
            )
        )
        M = euler_angles_to_matrix(thetas, "XZY").to(
            dtype=dtype
        )  # TODO need to support dtype in euler_angles_to_matrix
        return torch.matmul(M, unit.view(3, 1)).squeeze(-1)


def within_radius(radius: float, points: torch.Tensor) -> torch.Tensor:
    "Mask indicating if points of shape (..., 2|3) are within 'radius' distance from the X axis"
    
    dim = points.shape[-1]
    if dim == 2:
        r = points.select(-1, 1)
        return torch.le(torch.abs(r), radius)
    else:
        y, z = points.select(-1, 1), points.select(-1, 2)
        return torch.le(y**2 + z**2, radius**2)
