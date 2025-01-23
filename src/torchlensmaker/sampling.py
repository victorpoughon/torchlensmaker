import torch

from torchlensmaker.rot2d import rot2d
from torchlensmaker.rot3d import euler_angles_to_matrix

Tensor = torch.Tensor


def sample_line_linspace(N: int, diameter: float) -> Tensor:
    return torch.linspace(-diameter / 2, diameter / 2, N)


def sample_line_random(N: int, diameter: float) -> Tensor:
    return (torch.rand(N) - 0.5) * diameter


def sample_disk_random(N: int, diameter: float) -> tuple[Tensor, Tensor]:
    """
    Sample points on a disk using polar coordinates

    Args:
        N: number of samples
        diameter: diameter of the sampled disk

    Returns:
        P: tensor of the sampled 2D points
    """
    # Generate random r (square root for uniform distribution)
    r = torch.sqrt(torch.rand(N)) * (diameter / 2)

    # Generate random angles
    theta = torch.rand(N) * 2 * torch.pi

    # Convert polar coordinates to Cartesian coordinates
    X = r * torch.cos(theta)
    Y = r * torch.sin(theta)

    return torch.column_stack((X, Y))


def sample_disk_linspace(N: int, diameter: float) -> tuple[Tensor, Tensor]:
    """
    Sample points on a disk using polar coordinates with linspace distribution

    Args:
        N: number of samples (should be a perfect square for exactly N samples)
        diameter: diameter of the sampled disk

    Returns:
        P: tensor of the sampled 2D points
    """
    # Determine the grid size
    grid_size = int(torch.sqrt(torch.tensor(N)).floor())

    # Generate evenly spaced r and angles
    r = torch.linspace(0, diameter / 2, grid_size)
    theta = torch.linspace(0, 2 * torch.pi, grid_size)

    # Create a meshgrid of r and theta
    r, theta = torch.meshgrid(r, theta, indexing="ij")

    # Flatten the meshgrid
    r = r.flatten()
    theta = theta.flatten()

    # Convert polar coordinates to Cartesian coordinates
    X = r * torch.cos(theta)
    Y = r * torch.sin(theta)

    return torch.column_stack((X, Y))


def rotated_unit(angle1, angle2, dim, dtype):
    "Rotated X axis unit vector in degrees"

    angle1 = torch.deg2rad(angle1)
    angle2 = torch.deg2rad(angle2)

    # rays vectors
    if dim == 2:
        V = torch.tensor([1.0, 0.0], dtype=dtype)
        return rot2d(V, angle1)
    else:
        print("unit3")
        V = torch.tensor([1.0, 0.0, 0.0], dtype=dtype)
        M = euler_angles_to_matrix(
            torch.as_tensor([0.0, angle1, angle2], dtype=dtype),
            "XZY",
        ).to(
            dtype=dtype
        )  # TODO need to support dtype in euler_angles_to_matrix
        return V @ M.T
