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
import torch

from torchlensmaker.core.ray_bundle import RayBundle
from torchlensmaker.core.sampled_variable import SampledVariable


def _make_bundle_2d(n_pupil=3, n_field=2, n_wavel=2, source_idx=0):
    """Minimal 2D bundle with n_pupil*n_field*n_wavel rays, invariant-consistent."""
    N = n_pupil * n_field * n_wavel
    pupil_domain = torch.linspace(-1.0, 1.0, n_pupil)
    field_domain = torch.linspace(-0.5, 0.5, n_field)
    wavel_domain = torch.linspace(400.0, 700.0, n_wavel)

    # meshgrid indices: all combinations of pupil x field x wavel
    pi, fi, wi = torch.meshgrid(
        torch.arange(n_pupil), torch.arange(n_field), torch.arange(n_wavel),
        indexing="ij",
    )
    pupil_idx = pi.reshape(-1).to(torch.int64)
    field_idx = fi.reshape(-1).to(torch.int64)
    wavel_idx = wi.reshape(-1).to(torch.int64)

    return RayBundle.create(
        P=torch.zeros((N, 2)),
        V=torch.ones((N, 2)),
        pupil=SampledVariable.create(
            values=pupil_domain[pupil_idx],
            idx=pupil_idx,
            domain_values=pupil_domain,
            domain_idx=torch.arange(n_pupil, dtype=torch.int64),
        ),
        field=SampledVariable.create(
            values=field_domain[field_idx],
            idx=field_idx,
            domain_values=field_domain,
            domain_idx=torch.arange(n_field, dtype=torch.int64),
        ),
        wavel=SampledVariable.create(
            values=wavel_domain[wavel_idx],
            idx=wavel_idx,
            domain_values=wavel_domain,
            domain_idx=torch.arange(n_wavel, dtype=torch.int64),
        ),
        source=SampledVariable.create(
            values=torch.full((N,), float(source_idx)),
            idx=torch.full((N,), source_idx, dtype=torch.int64),
            domain_values=torch.tensor([float(source_idx)]),
            domain_idx=torch.tensor([source_idx], dtype=torch.int64),
        ),
    )


# --- empty ---

def test_empty_2d():
    b = RayBundle.empty(dim=2)
    assert b.P.shape == (0, 2)
    assert b.V.shape == (0, 2)
    assert b.pupil.values.shape == (0,)
    assert b.field.values.shape == (0,)
    assert b.wavel.values.shape == (0,)
    assert b.source.values.shape == (0,)
    assert b.pupil.domain_idx.shape == (0,)
    assert b.wavel.domain_idx.shape == (0,)


def test_empty_3d():
    b = RayBundle.empty(dim=3)
    assert b.P.shape == (0, 3)
    assert b.pupil.values.shape == (0, 2)
    assert b.field.values.shape == (0, 2)
    assert b.wavel.values.shape == (0,)


# --- mask ---

def test_mask_all_true_preserves_bundle():
    b = _make_bundle_2d(n_pupil=3, n_field=2, n_wavel=2)
    valid = torch.ones(b.P.shape[0], dtype=torch.bool)
    result = b.mask(valid)
    assert result.P.shape == b.P.shape
    assert torch.equal(result.pupil.domain_idx, b.pupil.domain_idx)
    assert torch.allclose(result.pupil.domain_values, b.pupil.domain_values)
    assert torch.equal(result.field.domain_idx, b.field.domain_idx)
    assert torch.allclose(result.field.domain_values, b.field.domain_values)
    assert torch.equal(result.wavel.domain_idx, b.wavel.domain_idx)
    assert torch.equal(result.source.domain_idx, b.source.domain_idx)


def test_mask_preserves_domain_after_full_filter():
    b = _make_bundle_2d(n_pupil=3, n_field=2, n_wavel=2)
    # Filter out all rays whose field_idx == 1
    valid = b.field.idx != 1
    result = b.mask(valid)
    # field_idx=1 should be filtered from values/idx but domain preserved
    assert 1 not in result.field.idx
    assert torch.equal(result.field.domain_idx, b.field.domain_idx)
    assert torch.allclose(result.field.domain_values, b.field.domain_values)


# --- cat ---

def test_cat_disjoint_sources():
    b0 = _make_bundle_2d(source_idx=0)
    b1 = _make_bundle_2d(source_idx=1)
    result = b0.cat(b1)
    assert result.P.shape[0] == b0.P.shape[0] + b1.P.shape[0]
    assert torch.equal(result.source.domain_idx, torch.tensor([0, 1], dtype=torch.int64))
    assert result.source.domain_values.shape == (2,)


def test_cat_conflicting_wavel_raises():
    b0 = _make_bundle_2d()
    # b1 with same wavel domain_idx=0 but different domain_value
    N = b0.P.shape[0]
    b1 = RayBundle.create(
        P=torch.zeros((N, 2)),
        V=torch.ones((N, 2)),
        pupil=b0.pupil,
        field=b0.field,
        wavel=SampledVariable.create(
            values=torch.full((N,), 600.0),
            idx=torch.zeros(N, dtype=torch.int64),
            domain_values=torch.tensor([600.0]),  # conflicts with b0's 400.0 at idx=0
            domain_idx=torch.tensor([0], dtype=torch.int64),
        ),
        source=b0.source,
    )
    with pytest.raises(AssertionError):
        b0.cat(b1)


def test_cat_empty_left():
    empty = RayBundle.empty(dim=2)
    b = _make_bundle_2d()
    result = empty.cat(b)
    assert result.P.shape == b.P.shape
    assert torch.equal(result.source.domain_idx, b.source.domain_idx)


def test_cat_empty_right():
    b = _make_bundle_2d()
    empty = RayBundle.empty(dim=2)
    result = b.cat(empty)
    assert result.P.shape == b.P.shape


# --- split_masks / split_by ---

def test_split_masks():
    b = _make_bundle_2d(n_pupil=3, n_field=2, n_wavel=2)
    N = b.P.shape[0]
    masks = b.split_masks("field")
    assert len(masks) == 2
    for m in masks:
        assert m.dtype == torch.bool
        assert m.shape == (N,)
    assert sum(m.sum().item() for m in masks) == N  # every ray in exactly one cell
    for k, m in enumerate(masks):
        assert (b.field.idx[m] == k).all()


def test_split_masks_combined_grid():
    """2D grid by computing both split_masks upfront and combining with &."""
    b = _make_bundle_2d(n_pupil=3, n_field=2, n_wavel=2)
    N = b.P.shape[0]
    sidecar = torch.arange(N, dtype=torch.float32)
    field_masks = b.split_masks("field")
    wavel_masks = b.split_masks("wavel")
    total = 0
    for ki, m1 in enumerate(field_masks):
        for kj, m2 in enumerate(wavel_masks):
            m = m1 & m2
            assert (b.field.idx[m] == ki).all()
            assert (b.wavel.idx[m] == kj).all()
            assert sidecar[m].shape == (m.sum(),)
            total += m.sum().item()
    assert total == N


def test_split_masks_chained_grid():
    """2D grid via two chained split_masks calls; sidecar stays aligned."""
    b = _make_bundle_2d(n_pupil=3, n_field=2, n_wavel=2)
    N = b.P.shape[0]
    sidecar = torch.arange(N, dtype=torch.float32)
    total = 0
    for ki, m1 in enumerate(b.split_masks("field")):
        sub = b.mask(m1)
        sub_sidecar = sidecar[m1]
        for kj, m2 in enumerate(sub.split_masks("wavel")):
            assert (sub.field.idx[m2] == ki).all()
            assert (sub.wavel.idx[m2] == kj).all()
            assert sub_sidecar[m2].shape == (m2.sum(),)
            total += m2.sum().item()
    assert total == N


def test_split_by():
    b = _make_bundle_2d(n_pupil=3, n_field=2, n_wavel=2)
    parts = b.split_by("field")
    assert len(parts) == 2  # one per field domain entry
    # all rays accounted for
    assert sum(p.P.shape[0] for p in parts) == b.P.shape[0]
    # each part contains only rays for its field idx
    for k, part in enumerate(parts):
        assert (part.field.idx == k).all()
    # domain preserved in every part
    for part in parts:
        assert torch.equal(part.field.domain_idx, b.field.domain_idx)
        assert torch.allclose(part.field.domain_values, b.field.domain_values)



def test_split_by_empty_cell_domain_preserved():
    b = _make_bundle_2d(n_pupil=3, n_field=2, n_wavel=2)
    parts = b.mask(b.field.idx != 1).split_by("field")
    # Both domain positions still present
    assert len(parts) == 2
    # Second cell is empty but domain intact
    assert parts[1].P.shape[0] == 0
    assert torch.equal(parts[1].field.domain_idx, b.field.domain_idx)
    assert torch.allclose(parts[1].field.domain_values, b.field.domain_values)


def test_split_by_unknown_var_raises():
    with pytest.raises(ValueError):
        _make_bundle_2d().split_by("unknown")


# --- domain invariant ---

def test_domain_invariant():
    """pupil.values[i] == pupil.domain_values[searchsorted(pupil.domain_idx, pupil.idx[i])]"""
    b = _make_bundle_2d(n_pupil=4, n_field=3, n_wavel=2)
    for sv in (b.pupil, b.field, b.wavel, b.source):
        pos = torch.searchsorted(sv.domain_idx, sv.idx)
        assert torch.allclose(sv.values, sv.domain_values[pos])
