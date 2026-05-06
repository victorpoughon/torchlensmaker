from functools import partial

import pytest
import torch

from torchimplicit.implicit_circle import (
    implicit_yzcircle_2d,
    implicit_yzcircle_3d,
)
from torchimplicit.implicit_disk import (
    implicit_disk_2d,
    implicit_disk_3d,
)
from torchimplicit.implicit_plane import (
    implicit_yaxis_2d,
    implicit_yzplane_3d,
)
from torchimplicit.implicit_sphere import (
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


def _make_2d_avoid_disk_boundary(R=1.5, margin=0.1):
    # Disk Hessian is discontinuous at |r| = R; exclude points near the boundary
    pts = _uniform(10000, 2)
    mask = (pts[:, 1].abs() - R).abs() < margin
    pts[mask] = torch.tensor([5.0, 5.0])
    return pts


def _make_3d_avoid_disk_boundary(R=1.5, margin=0.1):
    # Disk Hessian is discontinuous at sqrt(y^2+z^2) = R; exclude points near the boundary
    pts = _uniform(10000, 3)
    r = (pts[:, 1] ** 2 + pts[:, 2] ** 2).sqrt()
    mask = (r - R).abs() < margin
    pts[mask] = torch.tensor([5.0, 5.0, 5.0])
    return pts


# --- parametrized test cases ---

_R = torch.tensor([1.5])
_empty = torch.zeros(0)

cases_2d = [
    pytest.param(
        partial(implicit_yzcircle_2d, params=_R), _make_2d_avoid_r0, id="circle"
    ),
    pytest.param(
        partial(implicit_disk_2d, params=_R), _make_2d_avoid_disk_boundary, id="disk"
    ),
    pytest.param(
        partial(implicit_sphere_2d, params=_R), _make_2d_avoid_origin, id="sphere"
    ),
    pytest.param(
        partial(implicit_yaxis_2d, params=_empty), _make_2d_uniform, id="plane"
    ),
]

cases_3d = [
    pytest.param(
        partial(implicit_yzcircle_3d, params=_R), _make_3d_avoid_yz0, id="circle"
    ),
    pytest.param(
        partial(implicit_disk_3d, params=_R), _make_3d_avoid_disk_boundary, id="disk"
    ),
    pytest.param(
        partial(implicit_sphere_3d, params=_R), _make_3d_avoid_origin, id="sphere"
    ),
    pytest.param(
        partial(implicit_yzplane_3d, params=_empty), _make_3d_uniform, id="plane"
    ),
]
