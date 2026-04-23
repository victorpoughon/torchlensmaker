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

from collections import defaultdict
from typing import Any, DefaultDict

import torchlensmaker as tlm
from torchlensmaker.sequential.model_trace import ModelTrace


def test_trace_model() -> None:
    # Use half spheres to model interface boundaries
    radius = 5
    halfsphere = tlm.SphereByRadius(diameter=2 * radius, R=radius)

    optics = tlm.Sequential(
        # Position the light source just above the optical axis
        tlm.SubChain(
            tlm.Translate(y=5.001),
            tlm.ObjectAtInfinity(10, 0.5, wavelength=(400, 660)),
        ),
        # Move the droplet of water some distance away from the source
        tlm.Gap(50),
        # First interface: half sphere (pointing left), refractive air to water
        tlm.RefractiveSurface(
            halfsphere.clone(anchors=(1, 1)), materials=("air", "water")
        ),
        # Second interface: half sphere (pointing right), reflective
        tlm.SubChain(
            tlm.RotateMixed(-180),
            tlm.ReflectiveSurface(halfsphere.clone(anchors=(1, 1))),
        ),
        # Third interface: half sphere (pointing down), refractive water to air
        tlm.SubChain(
            tlm.RotateMixed(60),
            tlm.RefractiveSurface(
                halfsphere.clone(anchors=(1, 0)), materials=("water", "air")
            ),
        ),
    )

    trace = tlm.trace_model(optics, 2, tlm.SequentialData.empty(dim=2))

    assert isinstance(trace, ModelTrace)

    assert len(trace.input_rays) == 3
    assert len(trace.output_rays) == 8
    assert len(trace.input_joints) == 7
    assert len(trace.output_joints) == 7
    assert len(trace.surfaces) == 3
