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

from torchlensmaker.types import Tf2D, Tf3D, Tf

from torchlensmaker.kinematics.homogeneous_geometry import (
    hom_compose_2d,
    hom_compose_3d,
    hom_identity_2d,
    hom_identity_3d,
    hom_identity,
    hom_matrix_2d,
    hom_matrix_3d,
    hom_matrix,
    hom_scale,
    hom_scale_2d,
    hom_scale_3d,
    hom_rotate_2d,
    hom_rotate_3d,
    hom_translate_2d,
    hom_translate_3d,
    hom_translate,
    kinematic_chain_append,
    kinematic_chain_append_2d,
    kinematic_chain_append_3d,
    kinematic_chain_extend,
    kinematic_chain_extend_2d,
    kinematic_chain_extend_3d,
    transform_points,
    transform_rays,
    transform_vectors,
)

torch.set_default_dtype(torch.float64)
torch.set_default_device(torch.device("cpu"))


def test_hom_matrix() -> None:
    id2 = torch.eye(
        2, dtype=torch.get_default_dtype(), device=torch.get_default_device()
    )
    hom_id2 = hom_matrix(id2)
    assert torch.allclose(hom_id2, torch.eye(3, dtype=id2.dtype, device=id2.device))

    id3 = torch.eye(
        3, dtype=torch.get_default_dtype(), device=torch.get_default_device()
    )
    hom_id3 = hom_matrix(id3)
    assert torch.allclose(hom_id3, torch.eye(4, dtype=id2.dtype, device=id2.device))


def transforms_2d() -> list[Tf2D]:
    base = hom_identity_2d(
        dtype=torch.get_default_dtype(), device=torch.get_default_device()
    )
    t1 = hom_translate_2d(torch.tensor([1.0, 2.0]))
    t2 = hom_rotate_2d(torch.tensor(0.5))
    t3 = kinematic_chain_append_2d(base, t1)
    t4 = kinematic_chain_append_2d(t3, t2)
    t5 = kinematic_chain_append_2d(t4, t4)
    t6 = kinematic_chain_extend_2d(base, [t1, t2, t3])
    t7 = hom_identity_2d(
        dtype=torch.get_default_dtype(), device=torch.get_default_device()
    )
    t8 = hom_compose_2d([t1, t2, t3])
    t9 = hom_translate_2d(torch.tensor([0.5, 1.2]))
    t10 = hom_scale_2d(torch.tensor(1.0))
    t11 = hom_scale_2d(torch.tensor(-1.0))

    return [base, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11]


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
        for tf in transforms_2d():
            assert tf.dtype == points.dtype
            assert tf.dtype == vectors.dtype
            assert tf.device == points.device
            assert tf.device == vectors.device
            assert tf.shape == (3, 3)

            out_points = transform_points(tf.direct, points)
            assert out_points.dtype == points.dtype
            assert out_points.shape == points.shape
            assert out_points.device == points.device

            out_points_inv = transform_points(tf.inverse, points)
            assert out_points_inv.dtype == points.dtype
            assert out_points_inv.shape == points.shape
            assert out_points_inv.device == points.device

            out_vectors = transform_vectors(tf.direct, vectors)
            assert out_vectors.dtype == vectors.dtype
            assert out_vectors.shape == vectors.shape
            assert out_vectors.device == vectors.device

            out_vectors_inv = transform_vectors(tf.inverse, vectors)
            assert out_vectors_inv.dtype == vectors.dtype
            assert out_vectors_inv.shape == vectors.shape
            assert out_vectors_inv.device == vectors.device

            # Roundtrip
            assert torch.allclose(points, transform_points(tf.direct, out_points_inv))
            assert torch.allclose(points, transform_points(tf.inverse, out_points))
            assert torch.allclose(
                vectors, transform_vectors(tf.direct, out_vectors_inv)
            )
            assert torch.allclose(vectors, transform_vectors(tf.inverse, out_vectors))


def transforms_3d() -> list[Tf3D]:
    base = hom_identity_3d(
        dtype=torch.get_default_dtype(), device=torch.get_default_device()
    )
    t1 = hom_translate_3d(torch.tensor([1.0, 2.0, 3.0]))
    t2 = hom_rotate_3d(torch.tensor(0.5), torch.tensor(0.6))
    t3 = kinematic_chain_append_3d(base, t1)
    t4 = kinematic_chain_append_3d(t3, t2)
    t5 = kinematic_chain_append_3d(t4, t4)
    t6 = kinematic_chain_extend_3d(base, [t1, t2, t3])
    t7 = hom_identity_3d(
        dtype=torch.get_default_dtype(), device=torch.get_default_device()
    )
    t8 = hom_compose_3d([t1, t2, t3])
    t9 = hom_translate_3d(torch.tensor([1.0, 2.0, -3.0]))
    t10 = hom_scale_3d(torch.tensor(1.0))
    t11 = hom_scale_3d(torch.tensor(-1.0))

    return [base, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11]


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
        for tf in transforms_3d():
            assert tf.dtype == points.dtype
            assert tf.dtype == vectors.dtype
            assert tf.device == points.device
            assert tf.device == vectors.device
            assert tf.shape == (4, 4)

            out_points = transform_points(tf.direct, points)
            assert out_points.dtype == points.dtype
            assert out_points.shape == points.shape
            assert out_points.device == points.device

            out_points_inv = transform_points(tf.inverse, points)
            assert out_points_inv.dtype == points.dtype
            assert out_points_inv.shape == points.shape
            assert out_points_inv.device == points.device

            out_vectors = transform_vectors(tf.direct, vectors)
            assert out_vectors.dtype == vectors.dtype
            assert out_vectors.shape == vectors.shape
            assert out_vectors.device == vectors.device

            out_vectors_inv = transform_vectors(tf.inverse, vectors)
            assert out_vectors_inv.dtype == vectors.dtype
            assert out_vectors_inv.shape == vectors.shape
            assert out_vectors_inv.device == vectors.device

            out_rays_points, out_rays_vectors = transform_rays(
                tf.direct, points, vectors
            )
            assert torch.allclose(out_rays_points, out_points)
            assert torch.allclose(out_rays_vectors, out_vectors)

            # Roundtrip
            assert torch.allclose(points, transform_points(tf.direct, out_points_inv))
            assert torch.allclose(points, transform_points(tf.inverse, out_points))
            assert torch.allclose(
                vectors, transform_vectors(tf.direct, out_vectors_inv)
            )
            assert torch.allclose(vectors, transform_vectors(tf.inverse, out_vectors))
