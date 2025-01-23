import torch

Tensor = torch.Tensor


class SampleDisk:
    @staticmethod
    def sample(N: int, diameter: Tensor, dim: int) -> Tensor:
        if dim == 2:
            return sample_line_linspace(N, diameter)
        else:
            # careful this does not sample exactly N points if N in 3D when N is not a perfect square
            return sample_disk_linspace(N, diameter)


def sample_line_linspace(N: int, diameter: torch.Tensor) -> Tensor:
    return torch.linspace(-diameter / 2, diameter / 2, N, dtype=diameter.dtype)


def sample_line_random(N: int, diameter: torch.Tensor) -> Tensor:
    return (torch.rand(N, dtype=diameter.dtype) - 0.5) * diameter


def sample_disk_random(N: int, diameter: float) -> Tensor:
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


def sample_disk_linspace(N: int, diameter: Tensor) -> Tensor:
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
