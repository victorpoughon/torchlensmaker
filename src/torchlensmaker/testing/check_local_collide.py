import torch

import torchlensmaker as tlm


def check_local_collide(
    surface: tlm.LocalSurface, P: torch.Tensor, V: torch.Tensor, expected_collide: bool
) -> None:
    "Call surface.local_collide() and performs tests on the output"

    # Check that rays are the correct dtype
    assert P.dtype == surface.dtype
    assert V.dtype == surface.dtype

    # Call local_collide, rays in testing datasets are in local frame
    batch, D = P.shape[:-1], P.shape[-1]
    t, local_normals, valid = surface.local_collide(P, V)
    local_points = P + t.unsqueeze(-1).expand_as(V) * V

    # Check shapes
    assert t.dim() == len(batch) and t.shape == batch
    assert local_normals.dim() == len(batch) + 1 and local_normals.shape == (
        *batch,
        D,
    )
    assert valid.dim() == len(batch) and valid.shape == batch
    assert local_points.dim() == 2 and local_points.shape == (*batch, D)

    # Check dtypes
    assert t.dtype == surface.dtype, (P.dtype, V.dtype, t.dtype, surface.dtype)
    assert local_normals.dtype == surface.dtype
    assert valid.dtype == torch.bool
    assert local_points.dtype == surface.dtype

    # Check isfinite
    assert torch.all(torch.isfinite(t)), surface
    assert torch.all(torch.isfinite(local_normals))
    assert torch.all(torch.isfinite(valid))
    assert torch.all(torch.isfinite(local_points))

    # Check all normals are unit vectors
    assert torch.allclose(
        torch.linalg.vector_norm(local_normals, dim=-1),
        torch.ones(1, dtype=surface.dtype),
    )

    if isinstance(surface, tlm.ImplicitSurface):
        N = sum(P.shape[:-1])
        error = torch.sqrt(torch.sum(surface.Fd(local_points)**2) / N).item()
        rmse = surface.rmse(local_points)
    else:
        error = None
        rmse = None

    # Check expected collision against expected_collide
    assert torch.all(surface.contains(local_points) == expected_collide), (str(surface), error, rmse)
    assert torch.all(valid == expected_collide), surface

