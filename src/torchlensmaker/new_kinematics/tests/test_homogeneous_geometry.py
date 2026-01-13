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


import pytest
from typing import Any

import torch
import torch.nn as nn

from torchlensmaker.new_kinematics.homogeneous_geometry import (
    HomMatrix,
    hom_identity_2d,
    hom_identity_3d,
    hom_identity,
    hom_translate_2d,
    hom_translate_3d,
    hom_rotate_2d,
    hom_rotate_3d,
    transform_points,
    transform_vectors,
    transform_rays,
    kinematic_chain_append,
    kinematic_chain_extend,
)

torch.set_default_dtype(torch.float64)
torch.set_default_device(torch.device("cpu"))


def transforms_2d() -> list[tuple[HomMatrix, HomMatrix]]:
    base = hom_identity_2d(
        dtype=torch.get_default_dtype(), device=torch.get_default_device()
    )
    t1 = hom_translate_2d(torch.tensor([1.0, 2.0]))
    t2 = hom_rotate_2d(torch.tensor(0.5))
    t3 = kinematic_chain_append(*base, *t1)
    t4 = kinematic_chain_append(*t3, *t2)
    t5 = kinematic_chain_append(*t4, *t4)
    t6 = kinematic_chain_extend(*base, [t1[0], t2[0], t3[0]], [t1[1], t2[1], t3[1]])
    t7 = hom_identity(
        2, dtype=torch.get_default_dtype(), device=torch.get_default_device()
    )

    return [base, t1, t2, t3, t4, t5, t6, t7]


def test_transform_functions_2d() -> None:
    # Any number of batch dimensions
    test_points = [
        torch.rand((2,)),
        torch.rand((5, 2)),
        torch.rand((4, 5, 2)),
    ]

    test_vectors = [
        torch.rand((2,)),
        torch.rand((5, 2)),
        torch.rand((4, 5, 2)),
    ]

    for points, vectors in zip(test_points, test_vectors):
        for hom, hom_inv in transforms_2d():
            assert hom.dtype == points.dtype
            assert hom.dtype == vectors.dtype
            assert hom.device == points.device
            assert hom.device == vectors.device
            assert hom.shape == (3, 3)

            out_points = transform_points(hom, points)
            assert out_points.dtype == points.dtype
            assert out_points.shape == points.shape
            assert out_points.device == points.device

            out_points_inv = transform_points(hom_inv, points)
            assert out_points_inv.dtype == points.dtype
            assert out_points_inv.shape == points.shape
            assert out_points_inv.device == points.device

            out_vectors = transform_vectors(hom, vectors)
            assert out_vectors.dtype == vectors.dtype
            assert out_vectors.shape == vectors.shape
            assert out_vectors.device == vectors.device

            out_vectors_inv = transform_vectors(hom_inv, vectors)
            assert out_vectors_inv.dtype == vectors.dtype
            assert out_vectors_inv.shape == vectors.shape
            assert out_vectors_inv.device == vectors.device

            # Roundtrip
            assert torch.allclose(points, transform_points(hom, out_points_inv))
            assert torch.allclose(points, transform_points(hom_inv, out_points))
            assert torch.allclose(vectors, transform_vectors(hom, out_vectors_inv))
            assert torch.allclose(vectors, transform_vectors(hom_inv, out_vectors))


def transforms_3d() -> list[tuple[HomMatrix, HomMatrix]]:
    base = hom_identity_3d(
        dtype=torch.get_default_dtype(), device=torch.get_default_device()
    )
    t1 = hom_translate_3d(torch.tensor([1.0, 2.0, 3.0]))
    t2 = hom_rotate_3d(torch.tensor(0.5), torch.tensor(0.6))
    t3 = kinematic_chain_append(*base, *t1)
    t4 = kinematic_chain_append(*t3, *t2)
    t5 = kinematic_chain_append(*t4, *t4)
    t6 = kinematic_chain_extend(*base, [t1[0], t2[0], t3[0]], [t1[1], t2[1], t3[1]])
    t7 = hom_identity(
        3, dtype=torch.get_default_dtype(), device=torch.get_default_device()
    )

    return [base, t1, t2, t3, t4, t5, t6, t7]


def test_transform_functions_3d() -> None:
    # Any number of batch dimensions
    test_points = [
        torch.rand((3,)),
        torch.rand((5, 3)),
        torch.rand((4, 5, 3)),
    ]

    test_vectors = [
        torch.rand((3,)),
        torch.rand((5, 3)),
        torch.rand((4, 5, 3)),
    ]

    for points, vectors in zip(test_points, test_vectors):
        for hom, hom_inv in transforms_3d():
            assert hom.dtype == points.dtype
            assert hom.dtype == vectors.dtype
            assert hom.device == points.device
            assert hom.device == vectors.device
            assert hom.shape == (4, 4)

            out_points = transform_points(hom, points)
            assert out_points.dtype == points.dtype
            assert out_points.shape == points.shape
            assert out_points.device == points.device

            out_points_inv = transform_points(hom_inv, points)
            assert out_points_inv.dtype == points.dtype
            assert out_points_inv.shape == points.shape
            assert out_points_inv.device == points.device

            out_vectors = transform_vectors(hom, vectors)
            assert out_vectors.dtype == vectors.dtype
            assert out_vectors.shape == vectors.shape
            assert out_vectors.device == vectors.device

            out_vectors_inv = transform_vectors(hom_inv, vectors)
            assert out_vectors_inv.dtype == vectors.dtype
            assert out_vectors_inv.shape == vectors.shape
            assert out_vectors_inv.device == vectors.device

            out_rays_points, out_rays_vectors = transform_rays(hom, points, vectors)
            assert torch.allclose(out_rays_points, out_points)
            assert torch.allclose(out_rays_vectors, out_vectors)

            # Roundtrip
            assert torch.allclose(points, transform_points(hom, out_points_inv))
            assert torch.allclose(points, transform_points(hom_inv, out_points))
            assert torch.allclose(vectors, transform_vectors(hom, out_vectors_inv))
            assert torch.allclose(vectors, transform_vectors(hom_inv, out_vectors))
