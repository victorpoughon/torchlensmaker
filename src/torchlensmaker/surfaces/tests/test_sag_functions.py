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

from functools import partial

import pytest
import torch

from torchlensmaker.surfaces.sag_functions import (
    aspheric_sag_2d,
    aspheric_sag_3d,
    conical_sag_2d,
    conical_sag_3d,
    parabolic_sag_2d,
    parabolic_sag_3d,
    sag_sum_2d,
    sag_sum_3d,
    sag_to_implicit_2d_raw,
    spherical_sag_2d,
    spherical_sag_3d,
    xypolynomial_sag_3d,
)


def test_debug_shape_A():
    points = torch.rand((10, 1))

    g = spherical_sag_2d(points, C=torch.tensor(1 / 10))
    print(g.val.shape, g.grad.shape)


def test_debug_shape2b():
    points = torch.rand((10, 10, 2))
    sag = partial(spherical_sag_2d, C=torch.tensor(1 / 10))
    imp = sag_to_implicit_2d_raw(sag, nf=torch.tensor(1), tau=torch.tensor(1))

    F = imp(points, order=1)
    assert F.grad is not None
    print(F.val.shape, F.grad.shape)


def test_debug_shape2s():
    points = torch.rand((10, 1, 2))
    sag = partial(spherical_sag_2d, C=torch.tensor(1 / 10))
    imp = sag_to_implicit_2d_raw(sag, nf=torch.tensor(1), tau=torch.tensor(1))

    F = imp(points, order=1)
    assert F.grad is not None
    print(F.val.shape, F.grad.shape)


def test_debug_shape1b():
    points = torch.rand((10, 2))
    sag = partial(spherical_sag_2d, C=torch.tensor(1 / 10))
    imp = sag_to_implicit_2d_raw(sag, nf=torch.tensor(1), tau=torch.tensor(1))

    F = imp(points, order=1)
    assert F.grad is not None
    print(F.val.shape, F.grad.shape)


def test_debug_shape1s():
    points = torch.rand((1, 2))  # fails
    sag = partial(spherical_sag_2d, C=torch.tensor(1 / 10))
    imp = sag_to_implicit_2d_raw(sag, nf=torch.tensor(1), tau=torch.tensor(1))

    F = imp(points, order=1)
    assert F.grad is not None
    print(F.val.shape, F.grad.shape)


def test_debug_shape0():
    points = torch.rand((2))
    sag = partial(spherical_sag_2d, C=torch.tensor(1 / 10))
    imp = sag_to_implicit_2d_raw(sag, nf=torch.tensor(1), tau=torch.tensor(1))

    F = imp(points, order=1)
    assert F.grad is not None
    print(F.val.shape, F.grad.shape)


def test_sag_functions_2d() -> None:
    dtype, device = torch.float32, torch.device("cpu")

    sags = [
        partial(
            spherical_sag_2d,
            C=torch.tensor(1 / 15.0, dtype=dtype, device=device),
        ),
        partial(
            spherical_sag_2d,
            C=torch.tensor(1 / -15.0, dtype=dtype, device=device),
        ),
        partial(
            spherical_sag_2d,
            C=torch.tensor(0.0, dtype=dtype, device=device),
        ),
        partial(
            parabolic_sag_2d,
            A=torch.tensor(0.05, dtype=dtype, device=device),
        ),
        partial(
            parabolic_sag_2d,
            A=torch.tensor(-0.05, dtype=dtype, device=device),
        ),
        partial(
            parabolic_sag_2d,
            A=torch.tensor(0.0, dtype=dtype, device=device),
        ),
        partial(
            conical_sag_2d,
            C=torch.tensor(1 / 15.0, dtype=dtype, device=device),
            K=torch.tensor(-1.5, dtype=dtype, device=device),
        ),
        partial(
            conical_sag_2d,
            C=torch.tensor(1 / -15.0, dtype=dtype, device=device),
            K=torch.tensor(-1.5, dtype=dtype, device=device),
        ),
        partial(
            conical_sag_2d,
            C=torch.tensor(0.0, dtype=dtype, device=device),
            K=torch.tensor(-1.5, dtype=dtype, device=device),
        ),
        partial(
            conical_sag_2d,
            C=torch.tensor(1 / 15.0, dtype=dtype, device=device),
            K=torch.tensor(-1.0, dtype=dtype, device=device),
        ),
        partial(
            conical_sag_2d,
            C=torch.tensor(1 / -15.0, dtype=dtype, device=device),
            K=torch.tensor(-1.0, dtype=dtype, device=device),
        ),
        partial(
            conical_sag_2d,
            C=torch.tensor(0.0, dtype=dtype, device=device),
            K=torch.tensor(-1.0, dtype=dtype, device=device),
        ),
        partial(
            conical_sag_2d,
            C=torch.tensor(1 / 15.0, dtype=dtype, device=device),
            K=torch.tensor(-0.6, dtype=dtype, device=device),
        ),
        partial(
            conical_sag_2d,
            C=torch.tensor(1 / -15.0, dtype=dtype, device=device),
            K=torch.tensor(-0.6, dtype=dtype, device=device),
        ),
        partial(
            conical_sag_2d,
            C=torch.tensor(0.0, dtype=dtype, device=device),
            K=torch.tensor(-0.6, dtype=dtype, device=device),
        ),
        partial(
            conical_sag_2d,
            C=torch.tensor(1 / 15.0, dtype=dtype, device=device),
            K=torch.tensor(0.0, dtype=dtype, device=device),
        ),
        partial(
            conical_sag_2d,
            C=torch.tensor(1 / -15.0, dtype=dtype, device=device),
            K=torch.tensor(0.0, dtype=dtype, device=device),
        ),
        partial(
            conical_sag_2d,
            C=torch.tensor(0.0, dtype=dtype, device=device),
            K=torch.tensor(0.0, dtype=dtype, device=device),
        ),
        partial(
            conical_sag_2d,
            C=torch.tensor(1 / 15.0, dtype=dtype, device=device),
            K=torch.tensor(0.44, dtype=dtype, device=device),
        ),
        partial(
            conical_sag_2d,
            C=torch.tensor(1 / -15.0, dtype=dtype, device=device),
            K=torch.tensor(0.44, dtype=dtype, device=device),
        ),
        partial(
            conical_sag_2d,
            C=torch.tensor(0.0, dtype=dtype, device=device),
            K=torch.tensor(0.44, dtype=dtype, device=device),
        ),
        partial(
            aspheric_sag_2d,
            coefficients=torch.distributions.uniform.Uniform(-1.0, 1.0).sample((3,)),
        ),
        partial(
            sag_sum_2d,
            sags=[
                partial(
                    spherical_sag_2d,
                    C=torch.tensor(1 / 2.0, dtype=dtype, device=device),
                ),
                partial(
                    aspheric_sag_2d,
                    coefficients=torch.distributions.uniform.Uniform(-1.0, 1.0).sample((
                        3,
                    )),
                ),
            ],
        ),
    ]

    r_tensors = [
        torch.linspace(-1.0, 1.0, 100).reshape((100, 1)),
        torch.linspace(-1.0, 1.0, 100).reshape((10, 10, 1)),
        torch.linspace(-1.0, 1.0, 100).reshape((10, 2, 5, 1)),
        torch.linspace(-5.0, 5.0, 100).reshape(100, 1),
        torch.linspace(-5.0, 5.0, 100).reshape((10, 10, 1)),
        torch.linspace(-5.0, 5.0, 100).reshape((10, 2, 5, 1)),
        torch.linspace(-20.0, 20.0, 100).reshape((100, 1)),
    ]

    for sag in sags:
        for r in r_tensors:
            g = sag(r)
            assert torch.all(torch.isfinite(g.val))
            assert torch.all(torch.isfinite(g.grad))
            assert g.val.shape == r.shape[:-1]
            assert g.grad.shape == r.shape
            assert g.val.dtype == g.grad.dtype == r.dtype
            assert g.val.device == g.grad.device == r.device


def test_sag_functions_3d() -> None:
    dtype, device = torch.float32, torch.device("cpu")

    sags = [
        partial(
            spherical_sag_3d,
            C=torch.tensor(1 / 15.0, dtype=dtype, device=device),
        ),
        partial(
            spherical_sag_3d,
            C=torch.tensor(1 / -15.0, dtype=dtype, device=device),
        ),
        partial(
            spherical_sag_3d,
            C=torch.tensor(0.0, dtype=dtype, device=device),
        ),
        partial(
            parabolic_sag_3d,
            A=torch.tensor(0.05, dtype=dtype, device=device),
        ),
        partial(
            parabolic_sag_3d,
            A=torch.tensor(-0.05, dtype=dtype, device=device),
        ),
        partial(
            parabolic_sag_3d,
            A=torch.tensor(0.0, dtype=dtype, device=device),
        ),
        partial(
            conical_sag_3d,
            C=torch.tensor(1 / 15.0, dtype=dtype, device=device),
            K=torch.tensor(-1.5, dtype=dtype, device=device),
        ),
        partial(
            conical_sag_3d,
            C=torch.tensor(1 / -15.0, dtype=dtype, device=device),
            K=torch.tensor(-1.5, dtype=dtype, device=device),
        ),
        partial(
            conical_sag_3d,
            C=torch.tensor(0.0, dtype=dtype, device=device),
            K=torch.tensor(-1.5, dtype=dtype, device=device),
        ),
        partial(
            conical_sag_3d,
            C=torch.tensor(1 / 15.0, dtype=dtype, device=device),
            K=torch.tensor(-1.0, dtype=dtype, device=device),
        ),
        partial(
            conical_sag_3d,
            C=torch.tensor(1 / -15.0, dtype=dtype, device=device),
            K=torch.tensor(-1.0, dtype=dtype, device=device),
        ),
        partial(
            conical_sag_3d,
            C=torch.tensor(0.0, dtype=dtype, device=device),
            K=torch.tensor(-1.0, dtype=dtype, device=device),
        ),
        partial(
            conical_sag_3d,
            C=torch.tensor(1 / 15.0, dtype=dtype, device=device),
            K=torch.tensor(-0.6, dtype=dtype, device=device),
        ),
        partial(
            conical_sag_3d,
            C=torch.tensor(1 / -15.0, dtype=dtype, device=device),
            K=torch.tensor(-0.6, dtype=dtype, device=device),
        ),
        partial(
            conical_sag_3d,
            C=torch.tensor(0.0, dtype=dtype, device=device),
            K=torch.tensor(-0.6, dtype=dtype, device=device),
        ),
        partial(
            conical_sag_3d,
            C=torch.tensor(1 / 15.0, dtype=dtype, device=device),
            K=torch.tensor(0.0, dtype=dtype, device=device),
        ),
        partial(
            conical_sag_3d,
            C=torch.tensor(1 / -15.0, dtype=dtype, device=device),
            K=torch.tensor(0.0, dtype=dtype, device=device),
        ),
        partial(
            conical_sag_3d,
            C=torch.tensor(0.0, dtype=dtype, device=device),
            K=torch.tensor(0.0, dtype=dtype, device=device),
        ),
        partial(
            conical_sag_3d,
            C=torch.tensor(1 / 15.0, dtype=dtype, device=device),
            K=torch.tensor(0.44, dtype=dtype, device=device),
        ),
        partial(
            conical_sag_3d,
            C=torch.tensor(1 / -15.0, dtype=dtype, device=device),
            K=torch.tensor(0.44, dtype=dtype, device=device),
        ),
        partial(
            conical_sag_3d,
            C=torch.tensor(0.0, dtype=dtype, device=device),
            K=torch.tensor(0.44, dtype=dtype, device=device),
        ),
        partial(
            xypolynomial_sag_3d,
            coefficients=torch.distributions.uniform.Uniform(-1.0, 1.0).sample((3, 3)),
        ),
        partial(
            aspheric_sag_3d,
            coefficients=torch.distributions.uniform.Uniform(-1.0, 1.0).sample(
                (3,),
            ),
        ),
        partial(
            sag_sum_3d,
            sags=[
                partial(
                    spherical_sag_3d,
                    C=torch.tensor(1 / 2.0, dtype=dtype, device=device),
                ),
                partial(
                    xypolynomial_sag_3d,
                    coefficients=torch.distributions.uniform.Uniform(-1.0, 1.0).sample((
                        3,
                        3,
                    )),
                ),
            ],
        ),
    ]

    # Last dimension is (y,z)
    yz_tensors = [
        torch.distributions.uniform.Uniform(-1.0, 1.0).sample((10, 2)),
        torch.distributions.uniform.Uniform(-1.0, 1.0).sample((5, 5, 2)),
        torch.distributions.uniform.Uniform(-1.0, 1.0).sample((2, 4, 3, 2)),
        torch.distributions.uniform.Uniform(-5.0, 5.0).sample((10, 2)),
        torch.distributions.uniform.Uniform(-5.0, 5.0).sample((5, 5, 2)),
        torch.distributions.uniform.Uniform(-5.0, 5.0).sample((2, 4, 3, 2)),
        torch.distributions.uniform.Uniform(-20.0, 20.0).sample((2, 4, 3, 2)),
    ]

    for sag in sags:
        for yz in yz_tensors:
            g = sag(yz)

            assert torch.all(torch.isfinite(g.val))
            assert torch.all(torch.isfinite(g.grad))
            assert g.val.shape == yz[..., 0].shape
            assert g.grad.shape == yz.shape

            assert g.val.dtype == g.grad.dtype == yz.dtype
            assert g.val.device == g.grad.device == yz.device
