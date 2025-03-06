import pytest

from typing import Any, Iterable

import torch
import torch.nn as nn
import numpy as np

import torchlensmaker as tlm

from torchlensmaker.testing.collision_datasets import normal_rays

"""
Test all surfaces using the methods of the common base class LocalSurface().
All methods are tested here, except local_collide() which is only tested for the
basic stuff, without labeled data. A full test is in test_local_collide.
"""


@pytest.fixture(params=[2, 3], ids=["2D", "3D"])
def dim(request: pytest.FixtureRequest) -> Any:
    return request.param


@pytest.fixture(params=[torch.float32, torch.float64], ids=["float32", "float64"])
def dtype(request: pytest.FixtureRequest) -> Any:
    return request.param


@pytest.fixture
def surfaces(
    dtype: torch.dtype,
) -> list[tlm.LocalSurface]:
    # fmt: off
    return [
        tlm.CircularPlane(diameter=30, dtype=dtype),

        # Non axially symmetric surfaces are out of scope for now
        # tlm.SquarePlane(side_length=30, dtype=dtype),

        # Sphere
        tlm.Sphere(diameter=5, R=10, dtype=dtype),
        tlm.Sphere(diameter=5, C=0.05, dtype=dtype),
        tlm.Sphere(diameter=5, C=0., dtype=dtype),
        tlm.Sphere(diameter=5, R=tlm.parameter(10, dtype=dtype), dtype=dtype),
        tlm.Sphere(diameter=5, C=tlm.parameter(0.05, dtype=dtype), dtype=dtype),

        # SphereR
        tlm.SphereR(diameter=5, R=10, dtype=dtype),
        tlm.SphereR(diameter=5, C=0.05, dtype=dtype),
        tlm.SphereR(diameter=5, R=tlm.parameter(10, dtype=dtype), dtype=dtype),
        tlm.SphereR(diameter=5, C=tlm.parameter(0.05, dtype=dtype), dtype=dtype),

        # Half circle with SphereR
        tlm.SphereR(diameter=5, R=2.5, dtype=dtype),
        tlm.SphereR(diameter=5, R=tlm.parameter(2.5, dtype=dtype), dtype=dtype),

        # Parabola
        tlm.Parabola(diameter=5, A=0.05, dtype=dtype),
        tlm.Parabola(diameter=5, A=tlm.parameter(0.05, dtype=dtype), dtype=dtype),
        tlm.Parabola(diameter=5, A=-0.05, dtype=dtype),
        tlm.Parabola(diameter=5, A=tlm.parameter(-0.05, dtype=dtype), dtype=dtype),
        tlm.Parabola(diameter=5, A=0, dtype=dtype),
        tlm.Parabola(diameter=5, A=tlm.parameter(0, dtype=dtype), dtype=dtype),

        # Asphere
        tlm.Asphere(diameter=10, R=50, K=1.0, A4=0.005, dtype=dtype),
        tlm.Asphere(diameter=10, R=-50, K=1.0, A4=0.005, dtype=dtype),
        tlm.Asphere(diameter=10, R=50, K=1.0, A4=0.005, dtype=dtype),
        tlm.Asphere(diameter=10, R=50, K=-1.0, A4=0.005, dtype=dtype),
        tlm.Asphere(diameter=10, R=50, K=1.0, A4=0.005, dtype=dtype),
        tlm.Asphere(diameter=10, R=50, K=1.0, A4=-0.005, dtype=dtype),
        tlm.Asphere(diameter=10, R=50, K=1.0, A4=0.005, dtype=dtype),
        tlm.Asphere(diameter=10, R=-50, K=1.0, A4=0.005, dtype=dtype),
        tlm.Asphere(diameter=10, R=50, K=1.0, A4=0.005, dtype=dtype),
        tlm.Asphere(diameter=10, R=50, K=-1.0, A4=0.005, dtype=dtype),
        tlm.Asphere(diameter=10, R=50, K=1.0, A4=0.005, dtype=dtype),
        tlm.Asphere(diameter=10, R=50, K=1.0, A4=-0.005, dtype=dtype),
        tlm.Asphere(diameter=10, R=tlm.parameter(50, dtype=dtype), K=1.0, A4=0.005, dtype=dtype),
        tlm.Asphere(diameter=10, R=-50, K=tlm.parameter(1.0, dtype=dtype), A4=0.005, dtype=dtype),
        tlm.Asphere(diameter=10, R=50, K=1.0, A4=tlm.parameter(0.005, dtype=dtype), dtype=dtype),

        # TODO test domains of partial surfaces?
    ]
    # fmt: on


def test_testname(surfaces: list[tlm.LocalSurface]) -> None:
    for s in surfaces:
        assert len(s.testname()) > 0


def test_parameters(surfaces: list[tlm.LocalSurface]) -> None:
    for s in surfaces:
        params = s.parameters()
        assert isinstance(params, dict)
        assert all([isinstance(key, str) for key in params.keys()])
        assert all([isinstance(value, nn.Parameter) for value in params.values()])


def isflat(s: tlm.LocalSurface) -> bool:
    "True if the surface is completely flat"
    return (
        isinstance(s, tlm.Plane)
        or (
            isinstance(s, tlm.Sphere)
            and torch.allclose(s.inner_surface.C, torch.tensor(0.0, dtype=s.dtype))
        )
        or (
            isinstance(s, tlm.Parabola)
            and torch.allclose(s.A, torch.tensor(0.0, dtype=s.dtype))
        )
    )


def test_extent_and_zero(surfaces: list[tlm.LocalSurface]) -> None:
    for s in surfaces:
        zero2 = s.zero(2)
        extent2 = s.extent(2)
        assert zero2.dim() == 1 and zero2.shape == (2,)
        assert extent2.dim() == 1 and extent2.shape == (2,)
        assert torch.all(torch.isfinite(zero2))
        assert torch.all(torch.isfinite(extent2))
        assert zero2.dtype == s.dtype
        assert extent2.dtype == s.dtype, s.testname()
        assert torch.allclose(zero2, torch.tensor(0.0, dtype=s.dtype))
        if not isflat(s):
            assert not torch.allclose(extent2, torch.tensor(0.0, dtype=s.dtype))
        else:
            assert torch.allclose(extent2, torch.tensor(0.0, dtype=s.dtype))
        del zero2, extent2

        zero3 = s.zero(3)
        extent3 = s.extent(3)
        assert zero3.dim() == 1 and zero3.shape == (3,)
        assert extent3.dim() == 1 and extent3.shape == (3,)
        assert torch.all(torch.isfinite(zero3))
        assert torch.all(torch.isfinite(extent3))
        assert zero3.dtype == s.dtype
        assert extent3.dtype == s.dtype
        if not isflat(s):
            assert not torch.allclose(extent3, torch.tensor(0.0, dtype=s.dtype))
        else:
            assert torch.allclose(extent3, torch.tensor(0.0, dtype=s.dtype))
        del zero3, extent3


def sample_grid2d(lim: float, N: int, dtype: torch.dtype) -> torch.Tensor:
    x = np.linspace(-lim, lim, N)
    y = np.linspace(-lim, lim, N)
    X, Y = np.meshgrid(x, y)
    grid = torch.tensor(np.stack((X, Y), axis=-1).reshape(-1, 2))
    return grid.to(dtype=dtype)


def sample_grid3d(lim: float, N: int, dtype: torch.dtype) -> torch.Tensor:
    x = np.linspace(-lim, lim, N)
    y = np.linspace(-lim, lim, N)
    z = np.linspace(-lim, lim, N)
    X, Y, Z = np.meshgrid(x, y, z)
    grid = torch.tensor(np.stack((X, Y, Z), axis=-1).reshape(-1, 3))
    return grid.to(dtype=dtype)


def sample_grid(lim: float, N: int, dim: int, dtype: torch.dtype) -> torch.Tensor:
    if dim == 2:
        return sample_grid2d(lim=lim, N=N, dtype=dtype)
    else:
        return sample_grid3d(lim=lim, N=N, dtype=dtype)


def extra_batch_dims(tensor: torch.Tensor, dims: Iterable[int]) -> list[torch.Tensor]:
    "Create copies of tensor with extra batch dimensions"
    new_tensors = [tensor]
    for dim in dims:
        prev = new_tensors[-1]
        next = prev.unsqueeze(0).expand(dim, *([-1] * (prev.dim())))
        new_tensors.append(next)
    return new_tensors[1:]


def test_normals(surfaces: list[tlm.LocalSurface], dim: int) -> None:
    # Number of points per dimension of the sample grid
    # Make sure to use a sample grid with odd number of points so that 0 is
    # included
    N = 3

    # The sample grid gets reshaped to a single batch dimension
    # Number of points in the first batch dimension
    B1 = N**dim

    for s in surfaces:
        lim = 50  # TODO use 4*bbox.radial here instead of hardcoded limit
        points1 = sample_grid(lim, N, dim, dtype=s.dtype)

        # We're going to check that surface.normals() works with an arbitrary
        # number of batch dimensions.
        points2, points3, points4 = extra_batch_dims(points1, [4, 5, 6])

        assert points1.shape == (B1, dim)
        assert points2.shape == (4, B1, dim)
        assert points3.shape == (5, 4, B1, dim)
        assert points4.shape == (6, 5, 4, B1, dim)

        for points in (points1, points2, points3, points4):
            # Compute normals
            normals = s.normals(points)

            # Check shape
            assert normals.shape == points.shape

            # Check isfinite, dtype
            assert torch.all(torch.isfinite(normals))
            assert normals.dtype == s.dtype

            # Check vector norm along the last dimension is close to 1.0
            assert torch.allclose(
                torch.linalg.vector_norm(normals, dim=-1),
                torch.ones(normals.shape[:-1], dtype=s.dtype),
            )


def test_contains_and_samples2D(surfaces: list[tlm.LocalSurface]) -> None:
    N = 10
    epsilon = 1e-6
    # TODO float precision dependent epsilon and tolerance values

    for s in surfaces:
        samples_half = s.samples2D_half(N, epsilon)
        samples_full = s.samples2D_full(N, epsilon)

        for samples in (samples_half, samples_full):
            # Check dtype, shape and isfinite
            assert samples.dtype == s.dtype, s
            assert samples.shape == (N, 2)
            assert torch.all(samples.isfinite())

            # Check that samples are on the surface using the surface default tolerance
            contains_samples = s.contains(samples)
            assert torch.all(contains_samples)

            # Check shape and dtype of contains() mask
            assert contains_samples.shape == samples.shape[:-1]
            assert contains_samples.dtype == torch.bool

            # Check that samples that are sure not to be on the surface, are not
            modified_samples = (
                samples + 10 * s.extent(dim=2) + tlm.unit_vector(dim=2, dtype=s.dtype)
            )
            assert torch.all(torch.logical_not(s.contains(modified_samples)))

        # Make 3D samples by setting Z to zero
        # Check that they are on the surface
        samples3D = torch.column_stack(
            (samples_half, torch.zeros_like(samples_half[:, -1]))
        )
        assert torch.all(s.contains(samples3D))

        # Check that modified 3D samples are not on the surface
        modified_samples3D = (
            samples3D + 10 * s.extent(dim=3) + tlm.unit_vector(dim=3, dtype=s.dtype)
        )
        assert torch.all(torch.logical_not(s.contains(modified_samples3D)))

        # Check range
        pretty_much_positive = -1e-5
        assert torch.all(samples_half.select(-1, 1) >= pretty_much_positive)
        # TODO check upper range of samples_half with bbox
        # TODO check range of full with bbox


def test_local_collide_basic(surfaces: list[tlm.LocalSurface], dim: int) -> None:
    # Here we test all that we can about LocalSurface.local_collide(), for all
    # surfaces in the test cases of this test file. We only use a single dataset
    # of normal rays, which are expected to collide for every surface. More
    # advanded collision testing with more complex datasets is done in
    # test_local_collide.py
    gen = normal_rays(dim=dim, N=50, offset=10.0, epsilon=1e-3)

    for surface in surfaces:
        dataset = gen(surface)

        # Check that dataset uses surface dtype
        assert dataset.P.dtype == surface.dtype
        assert dataset.V.dtype == surface.dtype

        # Call local_collide, rays in testing datasets are in local frame
        P, V = dataset.P, dataset.V
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
        assert torch.all(torch.isfinite(t))
        assert torch.all(torch.isfinite(local_normals))
        assert torch.all(torch.isfinite(valid))
        assert torch.all(torch.isfinite(local_points))

        # Check all normals are unit vectors
        assert torch.allclose(
            torch.linalg.vector_norm(local_normals, dim=-1),
            torch.ones(1, dtype=surface.dtype),
        )

        # Normal rays are expected to collide for all surfaces
        assert torch.all(surface.contains(local_points)), surface
        assert torch.all(valid)

        # Rays and returned normals should be parallel, check dot product is close to one
        assert torch.allclose(
            torch.sum(V * local_normals, dim=-1),
            torch.ones(V.shape[:-1], dtype=V.dtype),
        )


def test_implicit_surface(surfaces: list[tlm.LocalSurface], dim: int) -> None:
    "Tests specific to implicit surfaces"

    # Number of points per dimension of the sample grid
    # Make sure to use a sample grid with odd number of points so that 0 is
    # included
    N = 3

    # The sample grid gets reshaped to a single batch dimension
    # Number of points in the first batch dimension
    B1 = N**dim

    for surface in [s for s in surfaces if isinstance(s, tlm.ImplicitSurface)]:
        lim = 50  # TODO use 4*bbox.radial here instead of hardcoded limit
        points1 = sample_grid(lim, N, dim, dtype=surface.dtype)

        # We're going to check that F and F_grad work with an arbitrary
        # number of batch dimensions.
        points2, points3, points4 = extra_batch_dims(points1, [4, 5, 6])

        assert points1.shape == (B1, dim)
        assert points2.shape == (4, B1, dim)
        assert points3.shape == (5, 4, B1, dim)
        assert points4.shape == (6, 5, 4, B1, dim)

        for points in (points1, points2, points3, points4):
            F = surface.Fd(points)
            F_grad = surface.Fd_grad(points)

            # Check shapes
            assert F.shape == points.shape[:-1]
            assert F_grad.shape == points.shape

            # Check isfinite, dtype
            assert torch.all(torch.isfinite(F))
            assert torch.all(torch.isfinite(F_grad))
            assert F.dtype == surface.dtype
            assert F_grad.dtype == surface.dtype
