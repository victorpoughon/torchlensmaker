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
from functools import partial

import torch

from torchlensmaker.implicit_surfaces.sag import (
    aspheric_sag_2d,
    aspheric_sag_3d,
    conical_sag_2d,
    conical_sag_3d,
    parabolic_sag_2d,
    parabolic_sag_3d,
    spherical_sag_2d,
    spherical_sag_3d,
)


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
    ]

    r_tensors = [
        torch.linspace(-1.0, 1.0, 100),
        torch.linspace(-1.0, 1.0, 100).reshape((10, 10)),
        torch.linspace(-1.0, 1.0, 100).reshape((10, 2, 5)),
    ]

    for sag in sags:
        for r in r_tensors:
            g, g_grad = sag(r)
            assert torch.all(torch.isfinite(g))
            assert torch.all(torch.isfinite(g_grad))
            assert g.shape == g_grad.shape == r.shape
            assert g.dtype == g_grad.dtype == r.dtype
            assert g.device == g_grad.device == r.device


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
    ]

    # Last dimension is (y,z)
    yz_tensors = [
        torch.distributions.uniform.Uniform(-1.0, 1.0).sample((10, 2)),
        torch.distributions.uniform.Uniform(-1.0, 1.0).sample((5, 5, 2)),
        torch.distributions.uniform.Uniform(-1.0, 1.0).sample((2, 4, 3, 2)),
    ]

    for sag in sags:
        for yz in yz_tensors:
            y, z = yz.unbind(-1)
            G, G_grad = sag(y, z)

            assert torch.all(torch.isfinite(G))
            assert torch.all(torch.isfinite(G_grad))
            assert G.shape == y.shape
            assert G_grad.shape[:-1] == y.shape
            assert G_grad.shape[-1] == 2

            assert G.dtype == G_grad.dtype == y.dtype
            assert G.device == G_grad.device == y.device
