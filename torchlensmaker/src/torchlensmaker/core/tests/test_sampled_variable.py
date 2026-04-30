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

import torchlensmaker as tlm
from torchlensmaker.core.sampled_variable import SampledVariable


def make_sv(values_list, idx_list, domain_values_list, domain_idx_list):
    return SampledVariable.create(
        values=torch.tensor(values_list),
        idx=torch.tensor(idx_list, dtype=torch.int64),
        domain_values=torch.tensor(domain_values_list),
        domain_idx=torch.tensor(domain_idx_list, dtype=torch.int64),
    )


# --- create validation ---


def test_create_mismatched_value_dtype():
    with pytest.raises(AssertionError):
        SampledVariable.create(
            values=torch.tensor([1.0], dtype=torch.float32),
            idx=torch.tensor([0], dtype=torch.int64),
            domain_values=torch.tensor([1.0], dtype=torch.float64),
            domain_idx=torch.tensor([0], dtype=torch.int64),
        )


def test_create_idx_wrong_dtype():
    with pytest.raises(AssertionError):
        SampledVariable.create(
            values=torch.tensor([1.0]),
            idx=torch.tensor([0], dtype=torch.int32),
            domain_values=torch.tensor([1.0]),
            domain_idx=torch.tensor([0], dtype=torch.int64),
        )


def test_create_domain_idx_wrong_dtype():
    with pytest.raises(AssertionError):
        SampledVariable.create(
            values=torch.tensor([1.0]),
            idx=torch.tensor([0], dtype=torch.int64),
            domain_values=torch.tensor([1.0]),
            domain_idx=torch.tensor([0], dtype=torch.int32),
        )


def test_create_mismatched_value_shape():
    with pytest.raises(AssertionError):
        SampledVariable.create(
            values=torch.zeros((3, 2), dtype=torch.float32),
            idx=torch.zeros(3, dtype=torch.int64),
            domain_values=torch.zeros((2, 3), dtype=torch.float32),
            domain_idx=torch.tensor([0, 1], dtype=torch.int64),
        )


def test_create_idx_shape_mismatch():
    with pytest.raises(AssertionError):
        SampledVariable.create(
            values=torch.tensor([1.0, 2.0]),
            idx=torch.tensor([0], dtype=torch.int64),  # wrong size
            domain_values=torch.tensor([1.0, 2.0]),
            domain_idx=torch.tensor([0, 1], dtype=torch.int64),
        )


def test_create_domain_idx_shape_mismatch():
    with pytest.raises(AssertionError):
        SampledVariable.create(
            values=torch.tensor([1.0]),
            idx=torch.tensor([0], dtype=torch.int64),
            domain_values=torch.tensor([1.0, 2.0]),
            domain_idx=torch.tensor([0], dtype=torch.int64),  # wrong size
        )


def test_create_domain_idx_not_sorted():
    with pytest.raises(AssertionError):
        SampledVariable.create(
            values=torch.tensor([1.0, 2.0, 3.0]),
            idx=torch.tensor([2, 0, 1], dtype=torch.int64),
            domain_values=torch.tensor([1.0, 2.0, 3.0]),
            domain_idx=torch.tensor([2, 0, 1], dtype=torch.int64),  # not sorted
        )


def test_create_domain_idx_not_unique():
    with pytest.raises(AssertionError):
        SampledVariable.create(
            values=torch.tensor([1.0, 2.0, 3.0]),
            idx=torch.tensor([0, 0, 1], dtype=torch.int64),
            domain_values=torch.tensor([1.0, 2.0]),
            domain_idx=torch.tensor([0, 0], dtype=torch.int64),  # not unique
        )


def test_create_valid_scalar():
    sv = make_sv([1.0, 2.0, 1.0], [0, 1, 0], [1.0, 2.0], [0, 1])
    assert sv.values.shape == (3,)
    assert sv.idx.shape == (3,)
    assert sv.domain_values.shape == (2,)
    assert sv.domain_idx.shape == (2,)


def test_create_valid_2d():
    sv = make_sv(
        [[1.0, 0.0], [0.0, 1.0]],
        [0, 1],
        [[1.0, 0.0], [0.0, 1.0]],
        [0, 1],
    )
    assert sv.values.shape == (2, 2)
    assert sv.domain_values.shape == (2, 2)


# --- empty ---


def test_empty_scalar():
    sv = SampledVariable.empty((), torch.float32, torch.device("cpu"))
    assert sv.values.shape == (0,)
    assert sv.idx.shape == (0,)
    assert sv.domain_values.shape == (0,)
    assert sv.domain_idx.shape == (0,)
    assert sv.values.dtype == torch.float32
    assert sv.idx.dtype == torch.int64


def test_empty_2d():
    sv = SampledVariable.empty((2,), torch.float64, torch.device("cpu"))
    assert sv.values.shape == (0, 2)
    assert sv.domain_values.shape == (0, 2)
    assert sv.values.dtype == torch.float64


# --- mask ---


def test_mask_filters_values_and_idx():
    sv = make_sv([1.0, 2.0, 3.0], [0, 1, 2], [1.0, 2.0, 3.0], [0, 1, 2])
    mask = torch.tensor([True, False, True])
    result = sv.mask(mask)
    assert torch.allclose(result.values, torch.tensor([1.0, 3.0]))
    assert torch.equal(result.idx, torch.tensor([0, 2], dtype=torch.int64))


def test_mask_preserves_domain():
    sv = make_sv([1.0, 2.0, 3.0], [0, 1, 2], [1.0, 2.0, 3.0], [0, 1, 2])
    mask = torch.tensor([True, False, False])
    result = sv.mask(mask)
    # Domain unchanged even though indices 1 and 2 are filtered
    assert torch.equal(result.domain_idx, sv.domain_idx)
    assert torch.allclose(result.domain_values, sv.domain_values)


def test_mask_all_false_preserves_domain():
    sv = make_sv([1.0, 2.0], [0, 1], [1.0, 2.0], [0, 1])
    mask = torch.tensor([False, False])
    result = sv.mask(mask)
    assert result.values.shape == (0,)
    assert torch.equal(result.domain_idx, sv.domain_idx)
    assert torch.allclose(result.domain_values, sv.domain_values)


# --- cat ---


def test_cat_disjoint_domains():
    sv_a = make_sv([1.0], [0], [1.0], [0])
    sv_b = make_sv([5.0], [5], [5.0], [5])
    result = sv_a.cat(sv_b)
    assert torch.equal(result.domain_idx, torch.tensor([0, 5], dtype=torch.int64))
    assert torch.allclose(result.domain_values, torch.tensor([1.0, 5.0]))
    assert torch.allclose(result.values, torch.tensor([1.0, 5.0]))
    assert torch.equal(result.idx, torch.tensor([0, 5], dtype=torch.int64))


def test_cat_overlapping_matching_domain():
    sv_a = make_sv([1.0, 2.0], [0, 1], [1.0, 2.0], [0, 1])
    sv_b = make_sv([1.0, 3.0], [0, 2], [1.0, 3.0], [0, 2])
    result = sv_a.cat(sv_b)
    # Domain should be union {0, 1, 2}
    assert torch.equal(result.domain_idx, torch.tensor([0, 1, 2], dtype=torch.int64))
    assert torch.allclose(result.domain_values, torch.tensor([1.0, 2.0, 3.0]))
    assert result.values.shape == (4,)


def test_cat_overlapping_mismatched_domain_raises():
    sv_a = make_sv([1.0], [0], [1.0], [0])
    sv_b = make_sv([9.0], [0], [9.0], [0])  # same idx=0 but different value
    with pytest.raises(AssertionError):
        sv_a.cat(sv_b)


def test_cat_empty_left():
    empty = SampledVariable.empty((), torch.float32, torch.device("cpu"))
    sv = make_sv([1.0, 2.0], [0, 1], [1.0, 2.0], [0, 1])
    result = empty.cat(sv)
    assert result is sv


def test_cat_empty_right():
    sv = make_sv([1.0, 2.0], [0, 1], [1.0, 2.0], [0, 1])
    empty = SampledVariable.empty((), torch.float32, torch.device("cpu"))
    result = sv.cat(empty)
    assert result is sv


def test_cat_2d_disjoint_domains():
    sv_a = make_sv([[1.0, 0.0]], [0], [[1.0, 0.0]], [0])
    sv_b = make_sv([[0.0, 1.0]], [5], [[0.0, 1.0]], [5])
    result = sv_a.cat(sv_b)
    assert torch.equal(result.domain_idx, torch.tensor([0, 5], dtype=torch.int64))
    assert result.domain_values.shape == (2, 2)
    assert result.values.shape == (2, 2)


# --- public API smoke test ---


def test_tlm_sampled_variable_import():
    assert tlm.SampledVariable is SampledVariable
