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

from torchlensmaker.sequential.sequential import ModelTrace
from torchlensmaker.viewer import tlmviewer


def trace_render_surfaces(scene: ModelTrace) -> list[Any]:
    surfaces = []
    for key, (tf, surface) in scene.surfaces.items():
        surf = surface.render()
        surf["matrix"] = tf.direct.tolist()
        surfaces.append(surf)

    return surfaces


def trace_render_joints(scene: ModelTrace) -> list[Any]:
    ret = []
    for tf in scene.joints.values():
        ret.extend(tlmviewer.render_joint(tf.direct))
    return ret


def render_model_trace(trace: ModelTrace) -> Any:
    "Render an OpticalScene to a tlmviewer scene"

    viewer_scene = tlmviewer.new_scene("2D" if trace.dim == 2 else "3D")

    # Render parts of the scene: surfaces, joints, rays
    viewer_scene["data"].extend(trace_render_surfaces(trace))
    viewer_scene["data"].extend(trace_render_joints(trace))

    return viewer_scene
