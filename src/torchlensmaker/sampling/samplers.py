import torch
import torch.nn as nn
import numpy as np

from typing import Any, Optional, Type, Sequence

Tensor = torch.Tensor

from torchlensmaker.tensor_manip import to_tensor

class Sampler:
    def size(self) -> int:
        return NotImplementedError

    def sample1d(self, diameter: Tensor, dtype: torch.dtype = torch.float64) -> Tensor:
        raise NotImplementedError

    def sample2d(self, diameter: Tensor, dtype: torch.dtype = torch.float64) -> Tensor:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}(size={repr(self.size())})"


class RandomUniformSampler(Sampler):
    def __init__(self, N: int):
        super().__init__()
        self.N = N

    def size(self) -> int:
        return self.N

    def sample1d(self, diameter: Tensor, dtype: torch.dtype = torch.float64) -> Tensor:
        return (torch.rand(self.N, dtype=dtype) - 0.5) * diameter

    def sample2d(self, diameter: Tensor, dtype: torch.dtype = torch.float64) -> Tensor:
        # Generate random r (square root for uniform distribution)
        r = torch.sqrt(torch.rand(self.N, dtype=dtype)) * (diameter / 2)

        # Generate random angles
        theta = torch.rand(self.N, dtype=dtype) * 2 * torch.pi

        # Convert polar coordinates to Cartesian coordinates
        X = r * torch.cos(theta)
        Y = r * torch.sin(theta)

        return torch.column_stack((X, Y))

class RandomNormalSampler(Sampler):
    def __init__(self, N: int, std: float):
        super().__init__()
        self.N = N
        self.std = std

    def size(self) -> int:
        return self.N

    def sample1d(self, diameter: Tensor, dtype: torch.dtype = torch.float64) -> Tensor:
        t = torch.zeros((self.N,), dtype=dtype)
        nn.init.trunc_normal_(t, mean=0.0, std=self.std, a=-diameter/2, b=diameter/2)
        return t

    def sample2d(self, diameter: Tensor, dtype: torch.dtype = torch.float64) -> Tensor:
        # Generate random r (square root for uniform distribution)
        t = torch.zeros((self.N,), dtype=dtype)
        nn.init.trunc_normal_(t, mean=0.0, std=self.std / (diameter / 2), a=0., b=1.)
        r = torch.sqrt(t) * (diameter / 2)

        # Generate random angles
        theta = torch.rand(self.N, dtype=dtype) * 2 * torch.pi

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


class DenseSampler(Sampler):
    def __init__(self, N: int):
        super().__init__()
        self.N = N

    def size(self) -> int:
        return self.N

    def sample1d(self, diameter: Tensor, dtype: torch.dtype = torch.float64) -> Tensor:
        return torch.linspace(-diameter / 2, diameter / 2, self.N, dtype=dtype)

    def sample2d(self, diameter: Tensor, dtype: torch.dtype = torch.float64) -> Tensor:
        return uniform_disk_sampling(self.N, diameter, dtype)


class ExactSampler(Sampler):
    def __init__(self, values: Sequence[float | int] | Tensor):
        self.values = to_tensor(values)
        if self.values.dim() == 1:
            self.values = self.values.unsqueeze(1)
        assert self.values.dim() == 2

    def size(self) -> int:
        return self.values.shape[0]
    
    def sample1d(self, _diameter: Tensor, dtype: torch.dtype = torch.float64) -> Tensor:
        assert self.values.shape[1] == 1, self.values.shape
        return self.values.to(dtype=dtype)

    def sample2d(self, _diameter: Tensor, dtype: torch.dtype = torch.float64) -> Tensor:
        assert self.values.shape[1] == 2
        return self.values.to(dtype=dtype)


def sampleND(
    sampler: Sampler,
    diameter: Tensor,
    dim: int,
    dtype: torch.dtype = torch.float64,
) -> Tensor:

    if dim == 2:
        return sampler.sample2d(diameter, dtype)
    elif dim == 1:
        return sampler.sample1d(diameter, dtype)
    else:
        raise RuntimeError(f"sampleND: dim must be 1 or 2, got {dim}")


def init_sampling(sampling: dict[str, Any]) -> dict[str, Sampler]:
    "Process a sampling dict and make Sampler objects from shortcut options"

    output: dict[str, Sampler] = {}
    for name, value in sampling.items():
        if isinstance(value, (float, int)):
            output[name] = DenseSampler(value)
        elif isinstance(value, (list, tuple)):
            output[name] = ExactSampler(value)
        else:
            output[name] = value
    
    return output


def dense(N: int) -> Sampler:
    "Dense sampling"

    return DenseSampler(N)


def random_uniform(N: int) -> Sampler:
    "Random uniform sampling"

    return RandomUniformSampler(N)


def random_normal(N: int, std: float) -> Sampler:
    "Random (truncated) normal sampling"

    return RandomNormalSampler(N, std)


def exact(values: Sequence[float | int] | Tensor) -> Sampler:
    "Exact sampling at the provided coordinates"

    return ExactSampler(values)
