import torch
import math
import numpy as np

from typing import Any, Optional, Type

Tensor = torch.Tensor


class Sampler:
    @staticmethod
    def sample1d(
        N: int, diameter: Tensor, dtype: torch.dtype = torch.float64
    ) -> Tensor:
        raise NotImplementedError

    @staticmethod
    def sample2d(
        N: int, diameter: Tensor, dtype: torch.dtype = torch.float64
    ) -> Tensor:
        raise NotImplementedError


def sampleND(
    name: Optional[str],
    N: int,
    diameter: Tensor,
    dim: int,
    dtype: torch.dtype = torch.float64,
) -> Tensor:
    sampler: Type[Sampler]
    if name is None:
        sampler = LinearDiskSampler
    else:
        sampler = {
            "random": RandomDiskSampler,
            "linear": LinearDiskSampler,
            "uniform": UniformDiskSampler,
        }[name]

    if dim == 2:
        return sampler.sample2d(N, diameter, dtype)
    elif dim == 1:
        return sampler.sample1d(N, diameter, dtype)
    else:
        raise RuntimeError(f"sampleND: dim must be 1 or 2, got {dim}")


class LinearDiskSampler(Sampler):
    @staticmethod
    def sample1d(
        N: int, diameter: Tensor, dtype: torch.dtype = torch.float64
    ) -> Tensor:
        return torch.linspace(-diameter / 2, diameter / 2, N, dtype=dtype)

    @staticmethod
    def sample2d(
        N: int, diameter: Tensor, dtype: torch.dtype = torch.float64
    ) -> Tensor:
        """
        Sample points on a disk using polar coordinates with linspace distribution.

        Careful this does not sample exactly N points when N is not a perfect square.

        Args:
            N: number of samples (should be a perfect square for exactly N samples)
            diameter: diameter of the sampled disk

        Returns:
            P: tensor of the sampled 2D points
        """
        # Determine the grid size
        grid_size = int(torch.sqrt(torch.tensor(N, dtype=dtype)).floor())

        # Generate evenly spaced r and angles
        r = torch.linspace(0, diameter / 2, grid_size, dtype=dtype)
        theta = torch.linspace(0, 2 * torch.pi, grid_size, dtype=dtype)

        # Create a meshgrid of r and theta
        r, theta = torch.meshgrid(r, theta, indexing="ij")

        # Flatten the meshgrid
        r = r.flatten()
        theta = theta.flatten()

        # Convert polar coordinates to Cartesian coordinates
        X = r * torch.cos(theta)
        Y = r * torch.sin(theta)

        return torch.column_stack((X, Y))


class RandomDiskSampler(Sampler):
    @staticmethod
    def sample1d(
        N: int, diameter: Tensor, dtype: torch.dtype = torch.float64
    ) -> Tensor:
        return (torch.rand(N, dtype=dtype) - 0.5) * diameter

    @staticmethod
    def sample2d(
        N: int, diameter: Tensor, dtype: torch.dtype = torch.float64
    ) -> Tensor:
        """
        Sample points randomly on a disk using polar coordinates

        Args:
            N: number of samples
            diameter: diameter of the sampled disk

        Returns:
            P: tensor of the sampled 2D points
        """
        # Generate random r (square root for uniform distribution)
        r = torch.sqrt(torch.rand(N, dtype=dtype)) * (diameter / 2)

        # Generate random angles
        theta = torch.rand(N, dtype=dtype) * 2 * torch.pi

        # Convert polar coordinates to Cartesian coordinates
        X = r * torch.cos(theta)
        Y = r * torch.sin(theta)

        return torch.column_stack((X, Y))


def uniform_disk_sampling(N: int, diameter: Tensor, dtype: torch.dtype) -> Tensor:
    M = np.floor((-np.pi + np.sqrt(np.pi**2 - 4 * np.pi * (1 - N))) / (2 * np.pi))
    if M == 0:
        M = 1
    alpha = (N - 1) / (np.pi * M * (M + 1))
    R = np.arange(1, M + 1)
    S = 2 * np.pi * alpha * R

    # If we're off, subtract the difference from the last element
    S = np.round(S)
    S[-1] -= S.sum() - (N - 1)
    S = S.astype(int)

    # List of sample points, start with the origin point
    points = [np.zeros((1, 2))]

    for s, r in zip(S, R):
        theta = np.linspace(-np.pi, np.pi, s + 1)[:-1]
        radius = r / M * diameter / 2
        points.append(np.column_stack((radius * np.cos(theta), radius * np.sin(theta))))

    return torch.tensor(np.vstack(points), dtype=dtype)


class UniformDiskSampler(Sampler):
    @staticmethod
    def sample1d(
        N: int, diameter: Tensor, dtype: torch.dtype = torch.float64
    ) -> Tensor:
        return torch.linspace(-diameter / 2, diameter / 2, N, dtype=dtype)

    @staticmethod
    def sample2d(
        N: int, diameter: Tensor, dtype: torch.dtype = torch.float64
    ) -> Tensor:
        return uniform_disk_sampling(N, diameter, dtype)
