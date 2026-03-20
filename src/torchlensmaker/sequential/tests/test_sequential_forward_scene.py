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

import torchlensmaker as tlm


def test_sequential_forward_scene() -> None:
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
            tlm.Rotate((-180, 0)),
            tlm.ReflectiveSurface(halfsphere.clone(anchors=(1, 1))),
        ),
        # Third interface: half sphere (pointing down), refractive water to air
        tlm.SubChain(
            tlm.Rotate((60, 0)),
            tlm.RefractiveSurface(
                halfsphere.clone(anchors=(1, 0)), materials=("water", "air")
            ),
        ),
    )

    scene = tlm.OpticalScene.empty()

    output = optics.forward_scene(tlm.SequentialData.empty(dim=2), "", scene)

    assert len(scene.rays) == 4
    assert len(scene.joints) == 7
    assert len(scene.surfaces) == 3
