import torch
import torch.nn as nn

from torchlensmaker.core.surfaces import ImplicitSurface, Sphere, Sphere3
from torchlensmaker.core.rot2d import perpendicular2d

Tensor = torch.Tensor

from dataclasses import dataclass


@dataclass
class CollisionDataset:
    """
    A collision dataset contains the data required for a single test case of collision detection.
    """

    name: str
    surface: ImplicitSurface
    expected_collide: bool
    P: Tensor
    V: Tensor


def offset_rays(
    surface: ImplicitSurface, offset: float, expected_collide: bool, N: int
):
    """
    Utility function to generate so called 'offset' rays: perpendicular to the
    surface gradient, offset along the gradient by a constant distance
    """

    samples = surface.samples2D_full(N)
    grad_samples = nn.functional.normalize(surface.f_grad(samples))
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
        expected_collide=expected_collide,
        P=P,
        V=V,
    )


def normal_rays(
    surface: ImplicitSurface, offset: float, expected_collide: bool, N: int
):
    """
    Utility function to generate so called 'normal' rays: parallel to the
    surface gradient, offset along the gradient by a constant distance
    """

    samples = surface.samples2D_full(N)
    grad_samples = nn.functional.normalize(surface.f_grad(samples))
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
        expected_collide=expected_collide,
        P=P,
        V=V,
    )


COLLISION_DATASET_REGISTRY = {}


def register_dataset(dataset) -> None:
    if dataset.name in COLLISION_DATASET_REGISTRY:
        raise RuntimeError(
            f"Dataset name {dataset.name} is already in collision dataset registry"
        )
    COLLISION_DATASET_REGISTRY[dataset.name] = dataset


register_dataset(
    normal_rays(
        surface=Sphere(10, 10),
        offset=5.0,
        expected_collide=True,
        N=15,
    )
)

register_dataset(
    normal_rays(
        surface=Sphere(10, 5),
        offset=1.0,
        expected_collide=True,
        N=150,
    )
)

register_dataset(
    normal_rays(
        surface=Sphere(10, 1e3),
        offset=5.0,
        expected_collide=True,
        N=15,
    )
)

register_dataset(
    normal_rays(
        surface=Sphere(10, 1e6),
        offset=5.0,
        expected_collide=True,
        N=15,
    )
)

register_dataset(
    normal_rays(
        surface=Sphere3(10, 10),
        offset=5.0,
        expected_collide=True,
        N=15,
    )
)

register_dataset(
    normal_rays(
        surface=Sphere3(10, 5),
        offset=1.0,
        expected_collide=True,
        N=15,
    )
)

register_dataset(
    offset_rays(
        surface=Sphere(10, 5),
        offset=1.0,
        expected_collide=False,
        N=15,
    )
)

register_dataset(
    offset_rays(
        surface=Sphere3(10, 5),
        offset=1.0,
        expected_collide=False,
        N=15,
    )
)

register_dataset(
    offset_rays(
        surface=Sphere3(10, 5),
        offset=-1.0,
        expected_collide=True,
        N=15,
    )
)
