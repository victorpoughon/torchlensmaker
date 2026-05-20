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
from torchlensmaker.materials.material_elements import NonDispersiveMaterial
from torchlensmaker.sequential.optical_path import linear_path


def _flat_slab_model(n_in: float, n_glass: float, gap1: float, thickness: float):
    """On-axis parallel rays through a flat glass slab."""
    return tlm.Sequential(
        tlm.ObjectAtInfinity(10, 0.5),
        tlm.Gap(gap1),
        tlm.RefractiveSurface(
            tlm.Plane(display_diameter=2.0),
            materials=(NonDispersiveMaterial(n_in), NonDispersiveMaterial(n_glass)),
        ),
        tlm.Gap(thickness),
        tlm.RefractiveSurface(
            tlm.Plane(display_diameter=2.0),
            materials=(NonDispersiveMaterial(n_glass), NonDispersiveMaterial(n_in)),
        ),
    )


def test_opl_shape():
    model = _flat_slab_model(1.0, 1.5, 50.0, 5.0)
    model.set_sampling2d(pupil=7, field=1, wavel=1)
    trace = tlm.raytrace(model, 2)

    path = linear_path(trace, "0", "4")
    opl = path.opl()

    assert opl.shape == (7,)
    assert (opl > 0).all()


def test_opl_flat_slab_no_aberration():
    # On-axis parallel rays through flat surfaces: all valid rays should have the
    # same OPL regardless of pupil position (no spherical or field aberration).
    n_in, n_glass = 1.0, 1.5
    model = _flat_slab_model(n_in, n_glass, 50.0, 5.0)
    model.set_sampling2d(pupil=11, field=1, wavel=1)
    trace = tlm.raytrace(model, 2)

    path = linear_path(trace, "0", "4")
    opl = path.opl()

    valid = trace.nodes["4"].bundle_out.valid
    opl_valid = opl[valid]

    assert opl_valid.numel() > 0
    # All valid rays should have identical OPL through flat on-axis geometry
    assert torch.allclose(opl_valid, opl_valid[0].expand_as(opl_valid), atol=1e-4)


def test_opl_gradients_flow():
    # OPL variance should be differentiable with respect to lens curvature.
    C = tlm.parameter(0.05)

    model = tlm.Sequential(
        tlm.ObjectAtInfinity(10, 0.5),
        tlm.Gap(50),
        tlm.RefractiveSurface(
            tlm.SphereByCurvature(diameter=10, C=C),
            materials=("air", "BK7"),
        ),
        tlm.Gap(5),
        tlm.RefractiveSurface(
            tlm.Plane(display_diameter=10),
            materials=("BK7", "air"),
        ),
    )

    model.set_sampling2d(pupil=11, field=1, wavel=1)
    trace = tlm.raytrace(model, 2)

    path = linear_path(trace, "0", "4")
    opl = path.opl()

    loss = opl.var()
    loss.backward()

    assert C.grad is not None
    assert C.grad.abs().item() > 0


def test_linear_path_key_errors():
    model = tlm.Sequential(
        tlm.ObjectAtInfinity(5, 0.5),
        tlm.Gap(50),
    )
    model.set_sampling2d(pupil=3, field=1, wavel=1)
    trace = tlm.raytrace(model, 2)

    with pytest.raises(KeyError):
        linear_path(trace, "missing_key", "1")

    with pytest.raises(KeyError):
        linear_path(trace, "0", "missing_key")


def test_opl_nested_subchain_matches_flat():
    # A surface wrapped in a SubChain should produce the same OPL as the
    # same surface placed directly in Sequential. Also verifies is_linear()
    # returns True for both topologies.

    def make_slab(first_surface):
        return tlm.Sequential(
            tlm.ObjectAtInfinity(10, 0.5),
            tlm.Gap(50),
            first_surface,
            tlm.Gap(5),
            tlm.RefractiveSurface(
                tlm.Plane(display_diameter=2.0),
                materials=(NonDispersiveMaterial(1.5), NonDispersiveMaterial(1.0)),
            ),
        )

    flat_surface = tlm.RefractiveSurface(
        tlm.Plane(display_diameter=2.0),
        materials=(NonDispersiveMaterial(1.0), NonDispersiveMaterial(1.5)),
    )
    nested_surface = tlm.SubChain(
        tlm.RefractiveSurface(
            tlm.Plane(display_diameter=2.0),
            materials=(NonDispersiveMaterial(1.0), NonDispersiveMaterial(1.5)),
        )
    )

    flat_model = make_slab(flat_surface)
    nested_model = make_slab(nested_surface)

    flat_model.set_sampling2d(pupil=11, field=1, wavel=1)
    nested_model.set_sampling2d(pupil=11, field=1, wavel=1)

    flat_trace = tlm.raytrace(flat_model, 2)
    nested_trace = tlm.raytrace(nested_model, 2)

    assert flat_trace.is_linear()
    assert nested_trace.is_linear()

    flat_opl = linear_path(flat_trace, "0", "4").opl()
    nested_opl = linear_path(nested_trace, "0", "4").opl()

    assert torch.allclose(flat_opl, nested_opl, atol=1e-5)
