import torch

from torchlensmaker.surfaces import LocalSurface
from torchlensmaker.transforms import TransformBase

Tensor = torch.Tensor


def intersect(
    surface: LocalSurface,
    P: Tensor,
    V: Tensor,
    transform: TransformBase,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Surface-rays collision detection

    Find collision points and normal vectors for the intersection of rays P+tV with
    a surface and a transform applied to that surface.

    Args:
        P: (N, 2|3) tensor, rays origins
        V: (N, 2|3) tensor, rays vectors
        surface: surface to collide with
        transform: transform applied to the surface

    Returns:
        points: valid collision points
        normals: valid surface normals at the collision points
        valid: bool tensor (N,) indicating which rays do collide with the surface
    """

    assert P.shape[0] == V.shape[0]
    assert P.shape[1] == V.shape[1]
    assert P.shape[1] in {2, 3}

    # Convert rays to surface local frame
    Ps = transform.inverse_points(P)
    Vs = transform.inverse_vectors(V)

    # Collision detection in the surface local frame
    t, local_normals, valid = surface.local_collide(Ps, Vs)

    # Compute collision points and convert normals to global frame
    points = P + t.unsqueeze(1).expand_as(V) * V
    normals = transform.direct_vectors(local_normals)

    # remove non valid (non intersecting) points
    # do this before computing global frame?
    points = points[valid]
    normals = normals[valid]

    # A surface always has two opposite normals, so keep the one pointing
    # against the ray, because that's what we need for refraction / reflection
    # i.e. the normal such that dot(normal, ray) < 0
    dot = torch.sum(normals * V[valid], dim=1)
    opposite_normals = torch.where(
        (dot > 0).unsqueeze(1).expand_as(normals), -normals, normals
    )

    assert points.shape == opposite_normals.shape
    assert valid.shape == (P.shape[0],)
    assert torch.all(torch.isfinite(points))
    assert torch.all(torch.isfinite(opposite_normals))
    assert torch.all(torch.isfinite(valid))

    return points, opposite_normals, valid
