import torch
import torch.nn as nn

from torchlensmaker.core.surfaces import ImplicitSurface, Sphere, Sphere3
from torchlensmaker.core.rot2d import perpendicular2d

Tensor = torch.Tensor

from dataclasses import dataclass, replace


@dataclass
class CollisionDataset:
    """
    A collision dataset contains the data required for a single test case of collision detection.
    """

    name: str
    surface: ImplicitSurface
    P: Tensor
    V: Tensor


def offset_rays(
    offset: float, N: int
):
    """
    Utility function to generate so called 'offset' rays: perpendicular to the
    surface gradient, offset along the gradient by a constant distance
    """

    def generate(surface: ImplicitSurface) -> CollisionDataset:
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


# surface
# rays generator(s)

# register_dataset(
#     normal_rays(
#         surface=Sphere(10, 10),
#         offset=5.0,
#         expected_collide=True,
#         N=15,
#     )
# )

# register_dataset(
#     normal_rays(
#         surface=Sphere(10, 5),
#         offset=1.0,
#         expected_collide=True,
#         N=150,
#     )
# )

# register_dataset(
#     normal_rays(
#         surface=Sphere(10, 1e3),
#         offset=5.0,
#         expected_collide=True,
#         N=15,
#     )
# )

# register_dataset(
#     normal_rays(
#         surface=Sphere(10, 1e6),
#         offset=5.0,
#         expected_collide=True,
#         N=15,
#     )
# )

# register_dataset(
#     normal_rays(
#         surface=Sphere3(10, 10),
#         offset=5.0,
#         expected_collide=True,
#         N=15,
#     )
# )

# register_dataset(
#     normal_rays(
#         surface=Sphere3(10, 5),
#         offset=1.0,
#         expected_collide=True,
#         N=15,
#     )
# )

# register_dataset(
#     offset_rays(
#         surface=Sphere(10, 5),
#         offset=1.0,
#         expected_collide=False,
#         N=15,
#     )
# )

# register_dataset(
#     offset_rays(
#         surface=Sphere3(10, 5),
#         offset=1.0,
#         expected_collide=False,
#         N=15,
#     )
# )

# register_dataset(
#     offset_rays(
#         surface=Sphere3(10, 5),
#         offset=-1.0,
#         expected_collide=True,
#         N=15,
#     )
# )
