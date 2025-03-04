import torch
import torch.nn as nn

from torchlensmaker.core.surfaces import ImplicitSurface, LocalSurface
from torchlensmaker.core.rot2d import perpendicular2d, rot2d
import math

Tensor = torch.Tensor

from dataclasses import dataclass, replace
from typing import Callable, TypeAlias


def make_samples(surface: LocalSurface, dim: int, N: int):
    "Generate samples and normals for a surface"

    if dim == 2:
        samples = surface.samples2D_full(N, epsilon=1e-3)
    else:
        root = math.ceil(math.sqrt(N))
        samples = make_samples3D(surface.samples2D_full(root, epsilon=1e-3), root)
    return samples, surface.normals(samples)


@dataclass
class CollisionDataset:
    name: str
    surface: LocalSurface
    P: Tensor
    V: Tensor


RayGenerator: TypeAlias = Callable[[LocalSurface], CollisionDataset]


def tangent_rays(dim: int, N: int, offset: float) -> RayGenerator:
    "Ray generator for rays tangent to the surface"

    # 2d only for now
    assert dim == 2

    def generate(surface: LocalSurface) -> CollisionDataset:
        samples, normals = make_samples(surface, dim, N)

        # Points on the surface, offset by 'offset' along the gradient
        P = samples + offset * normals

        # Vectors are perpendicular to the gradient
        V = nn.functional.normalize(perpendicular2d(normals))

        return CollisionDataset(
            f"{surface.testname()}_offset_{offset:.2f}", surface, P, V
        )

    return generate


def normal_rays(dim: int, N: int, offset: float) -> RayGenerator:
    "Ray generator for rays normal to the surface"

    def generate(surface: LocalSurface) -> CollisionDataset:
        samples, normals = make_samples(surface, dim, N)

        # offset along V
        P = samples + offset * normals
        V = normals

        return CollisionDataset(f"{surface.testname()}_normal", surface, P, V)

    return generate


def random_direction_rays(dim: int, N: int, offset: float) -> RayGenerator:
    "Ray generator for random direction rays"

    def generate(surface: LocalSurface) -> CollisionDataset:
        samples, _ = make_samples(surface, dim, N)

        theta = 2 * math.pi * torch.rand((N,))
        V = rot2d(torch.tensor([1.0, 0.0]), theta).to(dtype=surface.dtype)

        P = samples + offset * V

        return CollisionDataset(f"{surface.testname()}_random", surface, P, V)

    return generate


def fixed_rays(dim: int, N: int, direction: Tensor, offset: float) -> RayGenerator:
    "Ray generator for rays with a fixed direction"

    # normalize direction
    assert direction.numel() == dim
    direction = torch.nn.functional.normalize(direction, dim=0)

    def generate(surface: LocalSurface) -> CollisionDataset:
        samples, _ = make_samples(surface, dim, N)

        V = torch.tile(direction, (samples.shape[0], 1)).to(dtype=surface.dtype)
        P = samples + offset * V

        return CollisionDataset(f"{surface.testname()}_fixed", surface, P, V)

    return generate


def move_rays(P: Tensor, V: Tensor, m: float) -> tuple[Tensor, Tensor]:
    return P + m * V, V


def shift_dataset(dataset: CollisionDataset, shift: float) -> CollisionDataset:
    "Create a dataset with shifted rays"

    P, V = move_rays(dataset.P, dataset.V, shift)
    return replace(dataset, P=P, V=V)


def merge_datasets(datasets: list[CollisionDataset]) -> CollisionDataset:
    P = torch.cat([dataset.P for dataset in datasets], dim=0)
    V = torch.cat([dataset.V for dataset in datasets], dim=0)

    assert all([d.surface == datasets[0].surface for d in datasets])

    return CollisionDataset(
        name=f"merged-{len(datasets)}",
        surface=datasets[0].surface,
        P=P,
        V=V,
    )


def make_samples3D(samples2D: torch.Tensor, M: int) -> torch.Tensor:
    """
    Given a tensor of points of shape (N, 2), make a tensor of shape (M*N, 3) by
    axial rotation around the X axis and at M angles linearly spaced around the
    circle
    """
    step = 2 * torch.pi / M
    angles = torch.linspace(0, (M - 1) * step, M, device=samples2D.device)
    cosθ = torch.cos(angles)
    sinθ = torch.sin(angles)

    # Split coordinates and expand for broadcasting
    x = samples2D[:, 0].unsqueeze(1).repeat(1, M)  # (N, M)
    y = samples2D[:, 1].unsqueeze(1).repeat(1, M)  # (N, M)

    # Calculate rotated coordinates
    y_rotated = y * cosθ.unsqueeze(0)  # (N, M)
    z_rotated = y * sinθ.unsqueeze(0)  # (N, M)

    # Combine and reshape
    rotated_points = torch.stack([x, y_rotated, z_rotated], dim=2).view(-1, 3)

    return rotated_points
