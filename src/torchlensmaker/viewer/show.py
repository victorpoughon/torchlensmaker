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

import json
import uuid
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from torchlensmaker.core.base_module import BaseModule
from torchlensmaker.light_sources.light_sources_query import (
    set_sampling2d,
    set_sampling3d,
)
from torchlensmaker.sequential.model_trace import trace_model
from torchlensmaker.sequential.sequential_data import SequentialData
from torchlensmaker.viewer import tlmviewer as tlmviewer
from torchlensmaker.viewer.render_model_trace import render_model_trace


def inspect_stack(execute_list: list[tuple[nn.Module, Any, Any]]) -> None:
    for module, inputs, outputs in execute_list:
        print(type(module))
        print("inputs.transform:")
        for t in inputs.transforms:
            print(t)
        print()
        print("outputs.transform:")
        for t in outputs.transforms:
            print(t)
        print()


def render_model(
    optics: BaseModule,
    dim: int,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
    end: Optional[float] = None,
    title: str = "",
    controls: object | None = None,
) -> Any:
    "Render a model to tlmviewer scene"

    if dtype is None:
        dtype = torch.get_default_dtype()

    if device is None:
        device = torch.get_default_device()

    inputs = SequentialData.empty(dim=dim, dtype=dtype, device=device)
    trace = trace_model(optics, dim, inputs)
    scene = render_model_trace(optics, trace, end)

    if controls is not None:
        scene["controls"] = controls

    if title:
        scene["data"].append({"type": "scene-title", "title": title})

    return scene


def show(
    optics: BaseModule,
    dim: int = 2,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
    end: Optional[float] = None,
    title: str = "",
    ndigits: int | None = 8,
    controls: object | None = None,
    pupil: Any | None = None,
    field: Any | None = None,
    wavelength: Any | None = None,
) -> None:
    "Render an optical stack and show it with ipython display"

    if dim == 2:
        set_sampling2d(optics, pupil, field, wavelength)
    elif dim == 3:
        set_sampling3d(optics, pupil, field, wavelength)

    scene = render_model(optics, dim, dtype, device, end, title, controls)
    tlmviewer.display_scene(scene, ndigits)

    # debug: save all scenes
    # name = "tlmscene-" + str(uuid.uuid4())[:8] + ".json"
    # with open(name, "w") as f:
    #     json.dump(scene, f)


def show2d(*args, **kwargs):
    kwargs["dim"] = 2
    return show(*args, **kwargs)


def show3d(*args, **kwargs):
    kwargs["dim"] = 3
    return show(*args, **kwargs)


def export_json(
    optics: BaseModule,
    filename: str,
    dim: int = 2,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
    end: Optional[float] = None,
    title: str = "",
    ndigits: int | None = 8,
    controls: object | None = None,
    pupil: Any | None = None,
    field: Any | None = None,
    wavelength: Any | None = None,
) -> None:
    "Render and export an optical stack to a tlmviewer json file"

    if dim == 2:
        set_sampling2d(optics, pupil, field, wavelength)
    elif dim == 3:
        set_sampling3d(optics, pupil, field, wavelength)

    scene = render_model(optics, dim, dtype, device, end, title, controls)

    if ndigits is not None:
        scene = tlmviewer.truncate_scene(scene, ndigits)

    with open(filename, "w") as f:
        json.dump(scene, f)
