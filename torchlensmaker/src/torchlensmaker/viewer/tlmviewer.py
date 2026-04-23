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
import os.path
import string
import uuid
import warnings
from importlib.metadata import version
from pathlib import Path
from typing import Any, Optional

import tlmviewer as tlmv
import torch
from IPython.display import HTML, display

from torchlensmaker.kinematics.homogeneous_geometry import (
    HomMatrix,
    hom_identity,
    transform_points,
)
from torchlensmaker.surfaces import SurfaceElement

# Color theme
color_valid = "#ffa724"
color_blocked = "red"
color_spot_diagram = "coral"

Tensor = torch.Tensor


CATEGORY_VALID_RAYS = "rays-valid"
CATEGORY_BLOCKED_RAYS = "rays-blocked"
CATEGORY_OUTPUT_RAYS = "rays-output"
CATEGORY_JOINT = "kinematic-joint"


def get_script_template() -> str:
    "Get the js script template string"

    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(dir_path, "script_template.js"), encoding="utf-8") as f:
        return "<script type='module'>" + f.read() + "</script>"


def get_div_template() -> str:
    return "<div data-jp-suppress-context-menu id='$div_id' class='tlmviewer' style='width: 100%; aspect-ratio: 16 / 9;'></div>"


def random_id() -> str:
    return f"tlmviewer-{uuid.uuid4().hex[:8]}"


def pprint(scene: tlmv.Scene, ndigits: int | None = None) -> None:
    from pprint import pprint

    data = tlmv.scene_to_dict(scene)
    if ndigits is not None:
        json_data = json.dumps(data, allow_nan=False)
        data = json.loads(json_data, parse_float=lambda x: round(float(x), ndigits))

    pprint(data)


def dump(scene: tlmv.Scene, ndigits: int | None = None) -> None:
    data = tlmv.scene_to_dict(scene)
    if ndigits is not None:
        json_data = json.dumps(data, allow_nan=False)
        data = json.loads(json_data, parse_float=lambda x: round(float(x), ndigits))

    print(json.dumps(data))


def truncate_scene(scene: tlmv.Scene, ndigits: int) -> Any:
    json_data = json.dumps(tlmv.scene_to_dict(scene), allow_nan=False)
    return json.loads(json_data, parse_float=lambda x: round(float(x), ndigits))


def ipython_display(
    data: tlmv.Scene,
    ndigits: int | None = None,
) -> None:
    div_id = random_id()
    div_template = get_div_template()
    script_template = get_script_template()

    try:
        json_data = json.dumps(tlmv.scene_to_dict(data), allow_nan=False)
    except ValueError as err:
        warnings.warn(f"tlmviewer: got nan values in display data ({err})")
        json_data = json.dumps(tlmv.scene_to_dict(data), allow_nan=True)

    if ndigits is not None:
        json_data = json.dumps(
            json.loads(json_data, parse_float=lambda x: round(float(x), ndigits))
        )

    # Get torchlensmaker version from pyproject.toml metadata
    package_version = version("torchlensmaker")

    div = string.Template(div_template).substitute(div_id=div_id)
    script = string.Template(script_template).substitute(
        data=json_data,
        div_id=div_id,
        version=package_version,
    )
    display(HTML(div + script))  # type: ignore


vitepress_global_counter = 0


def vitepress_vue_display(
    data: tlmv.Scene,
    ndigits: int | None = None,
) -> None:
    global vitepress_global_counter

    name = os.environ["TLMVIEWER_TARGET_NAME"]
    target_dir = os.environ["TLMVIEWER_TARGET_DIRECTORY"]
    json_folder = Path(f"{name}_files")
    json_name = Path(f"{name}_{vitepress_global_counter}.json")

    output_folder = Path(target_dir) / json_folder
    output_folder.mkdir(parents=True, exist_ok=True)

    rel_path = json_folder / json_name
    abs_path = output_folder / Path(json_name)

    scene_dict = tlmv.scene_to_dict(data)
    if ndigits is not None:
        scene_dict = truncate_scene(data, ndigits)

    with open(abs_path, "w") as f:
        json.dump(scene_dict, f)

    # '?url' is required to workaround https://github.com/vitejs/vite-plugin-vue/issues/544
    # see also https://github.com/vuejs/vitepress/discussions/4619
    display(HTML(f'<TLMViewer src="./{rel_path}?url" />'))

    vitepress_global_counter += 1


def vue_format_requested() -> bool:
    return os.environ.get("TLMVIEWER_TARGET_FORMAT", None) == "vue"


def display_scene(scene: tlmv.Scene, ndigits: int | None = None) -> None:
    if not vue_format_requested():
        ipython_display(scene, ndigits)
    else:
        vitepress_vue_display(scene, ndigits)


def new_scene(mode: str) -> tlmv.Scene:
    if mode == "2D":
        return tlmv.Scene(mode="2D", camera="2D")
    elif mode == "3D":
        return tlmv.Scene(mode="3D", camera="orthographic")
    else:
        raise RuntimeError("mode should be 2D or 3D")


def render_surface(
    surface: SurfaceElement,
    dfk: HomMatrix,
    dim: int,
) -> Any:
    """
    Render a surface to a tlmviewer scene element
    """
    return surface.render(dfk)


def render_surface_local(
    surface: SurfaceElement,
    dim: int,
) -> Any:
    tfid = hom_identity(dim, dtype=torch.float32, device=torch.device("cpu"))
    return render_surface(
        surface,
        tfid.direct,
        dim=dim,
    )


def render_rays(
    start: Tensor,
    end: Tensor,
    category: str,
    variables: dict[str, Tensor] = {},
    domain: dict[str, list[float]] = {},
    default_color: str = "#ffa724",
) -> tlmv.Rays:
    assert start.shape == end.shape
    for var in variables.values():
        assert var.shape[0] == start.shape[0]

    points = torch.hstack((start, end)).tolist()
    variables_lists = {name: t.tolist() for name, t in variables.items()}
    dim = start.shape[-1]

    return tlmv.Rays(
        points=points,
        color=default_color,
        category=category,
        dim=dim,
        variables=variables_lists,
        domain=domain,
    )


def render_points(
    points: Tensor,
    color: str = "white",
    radius: Optional[float] = None,
    category: str = "",
) -> tlmv.Points:
    assert points.dim() == 2
    return tlmv.Points(
        data=points.tolist(),
        color=color,
        radius=radius if radius is not None else 0.0,
        category=category,
    )


def render_arrows(points: Tensor, normals: Tensor) -> tlmv.Arrows:
    return tlmv.Arrows(
        arrows=[n.tolist() + p.tolist() + [1.0] for p, n in zip(points, normals)]
    )


def render_collisions(points: Tensor, normals: Tensor) -> list[Any]:
    g1 = tlmv.Points(data=points.tolist(), color="#ff0000", radius=0.0, category="")
    g2 = tlmv.Arrows(
        arrows=[n.tolist() + p.tolist() + [1.0] for p, n in zip(points, normals)]
    )
    return [g1, g2]


def render_rays_until(
    P: Tensor,
    V: Tensor,
    end: Tensor,
    variables: dict[str, Tensor],
    domain: dict[str, list[float]],
    default_color: str,
    category: str,
) -> list[tlmv.Rays]:
    "Render rays until an absolute X coordinate"
    assert end.dim() == 0
    # div by zero here for vertical rays
    t = (end - P[:, 0]) / V[:, 0]
    ends = P + t.unsqueeze(1).expand_as(V) * V
    return [
        render_rays(
            P,
            ends,
            variables=variables,
            domain=domain,
            default_color=default_color,
            category=category,
        )
    ]


def render_rays_length(
    P: Tensor,
    V: Tensor,
    length: float | Tensor,
    variables: dict[str, Tensor],
    domain: dict[str, list[float]],
    category: str,
    default_color: str = color_valid,
) -> list[tlmv.Rays]:
    "Render rays with fixed length"

    if isinstance(length, Tensor):
        assert length.dim() in {0, 1}

    if isinstance(length, Tensor) and length.dim() == 1:
        length = length.unsqueeze(-1).expand_as(V)

    return [
        render_rays(
            P,
            P + length * V,
            variables=variables,
            domain=domain,
            default_color=default_color,
            category=category,
        )
    ]


def render_hit_miss_rays(
    P: torch.Tensor,
    V: torch.Tensor,
    t: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    variables_hit: dict[str, Tensor] = {},
    variables_miss: dict[str, Tensor] = {},
    domain: dict[str, list[float]] = {},
) -> list[Any]:
    collision_points = P + t.unsqueeze(1).expand_as(V) * V
    hits = valid.sum()
    misses = (~valid).sum()

    # Render hit rays
    rays_hit = (
        [
            render_rays(
                P[valid],
                collision_points[valid],
                variables=variables_hit,
                domain=domain,
                default_color=color_valid,
                category=CATEGORY_VALID_RAYS,
            )
        ]
        if hits > 0
        else []
    )

    # Render miss rays - rays absorbed because not colliding
    rays_miss = (
        render_rays_until(
            P[~valid],
            V[~valid],
            target,
            variables=variables_miss,
            domain=domain,
            default_color=color_blocked,
            category=CATEGORY_BLOCKED_RAYS,
        )
        if misses > 0
        else []
    )

    return rays_hit + rays_miss


def render_joint(dfk: HomMatrix) -> list[tlmv.Points]:
    dim, dtype = (
        dfk.shape[0] - 1,
        dfk.dtype,
    )

    origin = torch.zeros((dim,), dtype=dtype)
    joint = transform_points(dfk, origin)

    return [
        tlmv.Points(
            data=[joint.tolist()],
            color="white",
            radius=0.0,
            category=CATEGORY_JOINT,
        )
    ]
