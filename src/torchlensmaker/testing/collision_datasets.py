import torch
import torch.nn as nn

from torchlensmaker.core.surfaces import ImplicitSurface
from torchlensmaker.core.rot2d import perpendicular2d, rot2d
import math

Tensor = torch.Tensor

from dataclasses import dataclass, replace
from typing import Callable


@dataclass
class CollisionDataset:
    """
    A collision dataset contains the data required for a single test case of collision detection.
    """

    name: str
    surface: ImplicitSurface
    P: Tensor
    V: Tensor


def tangent_rays(
    offset: float, N: int
):
    """
    Utility function to generate so called 'tangent' rays: perpendicular to the
    surface gradient, offset along the gradient by a constant distance
    """

    def generate(surface: ImplicitSurface) -> CollisionDataset:
        samples = surface.samples2D_full(N, epsilon=1e-3)
        grad_samples = surface.normals(samples)
        assert torch.all(torch.isfinite(samples))
        assert torch.all(torch.isfinite(grad_samples))

        # Points on the surface, offset by 'offset' along the gradient
        P = samples + offset * grad_samples

        # Vectors are perpendicular to the gradient
        V = nn.functional.normalize(perpendicular2d(grad_samples))

        assert torch.all(torch.isfinite(P))
        assert torch.all(torch.isfinite(V))

        return CollisionDataset(
            name=f"{surface.testname()}_offset_{offset:.2f}",
            surface=surface,
            P=P,
            V=V,
        )

    return generate


def normal_rays(
    offset: float, N: int
):
    """
    Utility function to generate so called 'normal' rays: parallel to the
    surface gradient, offset along the gradient by a constant distance
    """

    def generate(surface: ImplicitSurface) -> CollisionDataset:

        samples = surface.samples2D_full(N, epsilon=1e-3)
        grad_samples = surface.normals(samples)
        assert torch.all(torch.isfinite(samples))
        assert torch.all(torch.isfinite(grad_samples))

        # Points on the surface, offset by 'offset' along the gradient
        P = samples + offset * grad_samples

        # Vectors are parallel to the gradient
        V = grad_samples

        assert torch.all(torch.isfinite(P))
        assert torch.all(torch.isfinite(V))

        return CollisionDataset(
            name=f"{surface.testname()}_normal",
            surface=surface,
            P=P,
            V=V,
        )

    return generate


def random_direction_rays(
    offset: float, N: int
):
    """
    Utility function to generate so called 'random direction' rays: 
    rays colliding with the surface on sample points with a random direction V
    offset along the gradient by a constant distance
    """

    def generate(surface: ImplicitSurface) -> CollisionDataset:

        samples = surface.samples2D_full(N, epsilon=1e-3)
        assert torch.all(torch.isfinite(samples))
        
        theta = 2 * math.pi * torch.rand((N,))
        V = rot2d(torch.tensor([1.0, 0.0]), theta).to(dtype=surface.dtype)

        P = samples + offset * V

        assert torch.all(torch.isfinite(P))
        assert torch.all(torch.isfinite(V))

        return CollisionDataset(
            name=f"{surface.testname()}_random",
            surface=surface,
            P=P,
            V=V,
        )

    return generate


def fixed_rays(
    direction: Tensor,
    offset: float,
    N: int,
) -> Callable[[ImplicitSurface], CollisionDataset]:
    """
    Ray generator for rays with a fixed direction
    """

    def generate(surface: ImplicitSurface) -> CollisionDataset:

        samples = surface.samples2D_full(N, epsilon=1e-3)
        assert torch.all(torch.isfinite(samples))
        
        V = torch.tile(direction, (N, 1)).to(dtype=surface.dtype)

        P = samples + offset * V

        assert torch.all(torch.isfinite(P))
        assert torch.all(torch.isfinite(V))

        return CollisionDataset(
            name=f"{surface.testname()}_fixed",
            surface=surface,
            P=P,
            V=V,
        )

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

