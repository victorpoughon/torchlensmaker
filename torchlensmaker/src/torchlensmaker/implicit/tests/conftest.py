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

from torchlensmaker.implicit.implicit_circle import (
    implicit_yzcircle_2d,
    implicit_yzcircle_3d,
)
from torchlensmaker.implicit.implicit_plane import (
    implicit_yaxis_2d,
    implicit_yzplane_3d,
)
from torchlensmaker.implicit.implicit_sphere import (
    implicit_sphere_2d,
    implicit_sphere_3d,
)


# --- point generators ---

def _uniform(n, d):
    return torch.distributions.uniform.Uniform(-20.0, 20.0).sample((n, d))


def _make_2d_uniform():
    return _uniform(10000, 2)


def _make_2d_avoid_r0():
    pts = _uniform(10000, 2)
    pts[:, 1] = pts[:, 1].clamp(min=0.2)
    return pts


def _make_2d_avoid_origin():
    pts = _uniform(10000, 2)
    mask = (pts**2).sum(dim=-1).sqrt() < 0.2
    pts[mask] = torch.tensor([0.5, 0.5])
    return pts


def _make_3d_uniform():
    return _uniform(10000, 3)


def _make_3d_avoid_yz0():
    pts = _uniform(10000, 3)
    pts[:, 1:] = pts[:, 1:].clamp(min=0.2)
    return pts


def _make_3d_avoid_origin():
    pts = _uniform(10000, 3)
    mask = (pts**2).sum(dim=-1).sqrt() < 0.2
    pts[mask] = torch.tensor([0.5, 0.5, 0.5])
    return pts


# --- parametrized test cases ---

cases_2d = [
    pytest.param(partial(implicit_yzcircle_2d, R=1.5), _make_2d_avoid_r0, id="circle"),
    pytest.param(partial(implicit_sphere_2d, R=1.5), _make_2d_avoid_origin, id="sphere"),
    pytest.param(partial(implicit_yaxis_2d), _make_2d_uniform, id="plane"),
]

cases_3d = [
    pytest.param(partial(implicit_yzcircle_3d, R=1.5), _make_3d_avoid_yz0, id="circle"),
    pytest.param(partial(implicit_sphere_3d, R=1.5), _make_3d_avoid_origin, id="sphere"),
    pytest.param(partial(implicit_yzplane_3d), _make_3d_uniform, id="plane"),
]
