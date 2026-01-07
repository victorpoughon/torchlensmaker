# This file is part of Torch Lens Maker
# Copyright (C) 2024-present Victor Poughon
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from typing import Iterable

import torch
import torch.nn as nn
import numpy as np

import torchlensmaker as tlm

from torchlensmaker.testing.check_local_collide import check_local_collide
from torchlensmaker.testing.collision_datasets import NormalRays
from torchlensmaker.core.geometry import sample_bcyl

"""
Test all surfaces using the methods of the common base class LocalSurface().
All methods are tested here, except local_collide() which is only tested for the
basic stuff, without labeled data. A full test is in test_local_collide.
"""


def test_todict(surfaces: list[tlm.LocalSurface]) -> None:
    for s in surfaces:
        assert len(s.to_dict(2)) > 0
        assert len(s.to_dict(3)) > 0


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
            and torch.allclose(s.sag_function.C, torch.tensor(0.0, dtype=s.dtype))
        )
        or (
            isinstance(s, tlm.Parabola)
            and torch.allclose(s.sag_function.A, torch.tensor(0.0, dtype=s.dtype))
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
        assert extent2.dtype == s.dtype, s.to_dict()
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


def extra_batch_dims(tensor: torch.Tensor, dims: Iterable[int]) -> list[torch.Tensor]:
    "Create copies of tensor with extra batch dimensions"
    new_tensors = [tensor]
    for dim in dims:
        prev = new_tensors[-1]
        next = prev.unsqueeze(0).expand(dim, *([-1] * (prev.dim())))
        new_tensors.append(next)
    return new_tensors[1:]


def test_normals(surfaces: list[tlm.LocalSurface], dim: int) -> None:
    "Test normals at sample points locations"

    # TODO enable this test for the 3D non symmetric case
    # need to add a surface.samples3d function
    if dim == 3:
        return

    N = 10

    for s in surfaces:
        points1 = s.samples2D_full(N, epsilon=0.01)
        B1 = N

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
    gen = NormalRays(dim=dim, N=50, offset=10.0, epsilon=1e-2)

    for surface in surfaces:
        P, V = gen(surface)
        check_local_collide(surface, P, V, expected_collide=True)


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
        points0 = torch.zeros((dim,), dtype=surface.dtype)
        xmin, xmax, tau = surface.bcyl()
        points1 = sample_bcyl(N, xmin, xmax, tau, dim, dtype=surface.dtype)

        # We're going to check that F and F_grad work with an arbitrary
        # number of batch dimensions.
        points2, points3, points4 = extra_batch_dims(points1, [4, 5, 6])

        assert points0.shape == (dim,)
        assert points1.shape == (B1, dim)
        assert points2.shape == (4, B1, dim)
        assert points3.shape == (5, 4, B1, dim)
        assert points4.shape == (6, 5, 4, B1, dim)

        for points in (points0, points1, points2, points3, points4):
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
