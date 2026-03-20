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


from typing import Any

import torchlensmaker as tlm


def scene_render_surfaces(scene: tlm.OpticalScene) -> Any:
    surfaces = []
    for key, (tf, surface) in scene.surfaces.items():
        surf = surface.render()
        surf["matrix"] = tf

    return surfaces


def render_scene(scene: tlm.OpticalScene) -> Any:
    "Render an OpticalScene to a tlmviewer scene"

    # render rays
    # render joints
    # render surfaces
