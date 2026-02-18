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

from IPython.display import display, HTML
import string
import uuid
import os.path
import json
import torch

import warnings
from pathlib import Path

from importlib.metadata import version

from typing import Any, Optional

from torchlensmaker.surfaces.local_surface import LocalSurface
from torchlensmaker.surfaces.implicit_surface import ImplicitSurface

from torchlensmaker.kinematics.homogeneous_geometry import (
    HomMatrix,
    hom_identity,
    transform_points,
)

# Color theme
color_valid = "#ffa724"
color_blocked = "red"
color_spot_diagram = "coral"

Tensor = torch.Tensor


LAYER_VALID_RAYS = 1
LAYER_BLOCKED_RAYS = 2
LAYER_OUTPUT_RAYS = 3
LAYER_JOINTS = 4


def get_script_template() -> str:
    "Get the js script template string"

    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(dir_path, "script_template.js"), encoding="utf-8") as f:
        return "<script type='module'>" + f.read() + "</script>"


def get_div_template() -> str:
    return "<div data-jp-suppress-context-menu id='$div_id' class='tlmviewer' style='width: 100%; aspect-ratio: 16 / 9;'></div>"


def random_id() -> str:
    return f"tlmviewer-{uuid.uuid4().hex[:8]}"


def pprint(scene: object, ndigits: int | None = None) -> None:
    from pprint import pprint

    if ndigits is not None:
        json_data = json.dumps(scene, allow_nan=False)
        scene = json.loads(json_data, parse_float=lambda x: round(float(x), ndigits))

    pprint(scene)


def dump(scene: object, ndigits: int | None = None) -> None:
    if ndigits is not None:
        json_data = json.dumps(scene, allow_nan=False)
        scene = json.loads(json_data, parse_float=lambda x: round(float(x), ndigits))

    print(json.dumps(scene))


def truncate_scene(scene, ndigits: int) -> Any:
    json_data = json.dumps(scene, allow_nan=False)
    scene = json.loads(json_data, parse_float=lambda x: round(float(x), ndigits))

    return scene


def ipython_display(
    data: object,
    ndigits: int | None = None,
) -> None:
    div_id = random_id()
    div_template = get_div_template()
    script_template = get_script_template()

    try:
        json_data = json.dumps(data, allow_nan=False)
    except ValueError as err:
        warnings.warn(f"tlmviewer: got nan values in display data ({err})")
        json_data = json.dumps(data, allow_nan=True)

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
    data: object,
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

    # truncate digits if requested
    if ndigits is not None:
        data = truncate_scene(data, ndigits)

    # write json file
    with open(abs_path, "w") as f:
        json.dump(data, f)

    # '?url' is required to workaround https://github.com/vitejs/vite-plugin-vue/issues/544
    # see also https://github.com/vuejs/vitepress/discussions/4619
    display(HTML(f'<TLMViewer src="./{rel_path}?url" />'))

    vitepress_global_counter += 1


def vue_format_requested() -> bool:
    return os.environ.get("TLMVIEWER_TARGET_FORMAT", None) == "vue"


def display_scene(scene: object, ndigits: int | None = None) -> None:
    if not vue_format_requested():
        ipython_display(scene, ndigits)
    else:
        vitepress_vue_display(scene, ndigits)


def new_scene(mode: str) -> Any:
    if mode == "2D":
        return {"mode": "2D", "camera": "XY", "data": []}
    elif mode == "3D":
        return {"mode": "3D", "camera": "orthographic", "data": []}
    else:
        raise RuntimeError("mode should be 2D or 3D")


def render_surface(
    surface: ImplicitSurface,
    dfk: HomMatrix,
    dim: int,
) -> object:
    """
    Render a surface to a json serializable object in tlmviewer format
    """

    # Convert the surface to a dict
    obj = surface.to_dict(dim)

    # Add the matrix transform
    obj["matrix"] = dfk.tolist()

    return obj


def render_surface_local(
    surface: LocalSurface,
    dim: int,
) -> Any:
    tfid = hom_identity(
        dim, dtype=surface.dtype, device=torch.device("cpu")
    )  # TODO gpu support
    return render_surface(
        surface,
        tfid.direct,
        dim=dim,
    )


def render_rays(
    start: Tensor,
    end: Tensor,
    layer: int,
    variables: dict[str, Tensor] = {},
    domain: dict[str, list[float]] = {},
    default_color: str = "#ffa724",
) -> Any:
    assert start.shape == end.shape
    for var in variables.values():
        assert var.shape[0] == start.shape[0]

    points = torch.hstack((start, end)).tolist()

    variables_lists = {name: t.tolist() for name, t in variables.items()}

    node = {
        "type": "rays",
        "points": points,
        "color": default_color,
        "variables": variables_lists,
        "domain": domain,
    }

    if layer is not None:
        node.update(layers=[layer])

    return node


def render_points(
    points: Tensor, color: str = "white", radius: Optional[float] = None
) -> Any:
    # TODO option to render points sizes in screen coordinates
    assert points.dim() == 2
    return {
        "type": "points",
        "data": points.tolist(),
        "color": color,
        **({"radius": radius} if radius is not None else {}),
    }


def render_collisions(points: Tensor, normals: Tensor) -> Any:
    g1 = {
        "type": "points",
        "data": points.tolist(),
        "color": "#ff0000",
    }

    g2 = {
        "type": "arrows",
        "data": [n.tolist() + p.tolist() + [1.0] for p, n in zip(points, normals)],
    }

    return [g1, g2]


def render_rays_until(
    P: Tensor,
    V: Tensor,
    end: Tensor,
    variables: dict[str, Tensor],
    domain: dict[str, list[float]],
    default_color: str,
    layer: int,
) -> list[Any]:
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
            layer=layer,
        )
    ]


def render_rays_length(
    P: Tensor,
    V: Tensor,
    length: float | Tensor,
    variables: dict[str, Tensor],
    domain: dict[str, list[float]],
    layer: int,
    default_color: str = color_valid,
) -> list[Any]:
    "Render rays with fixed length"

    if isinstance(length, Tensor):
        assert length.dim() in {0, 1}

    if isinstance(length, Tensor) and length.dim() == 1:
        length = length.unsqueeze(1).expand_as(V)

    return [
        render_rays(
            P,
            P + length * V,
            variables=variables,
            domain=domain,
            default_color=default_color,
            layer=layer,
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
) -> Any:
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
                layer=LAYER_VALID_RAYS,
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
            layer=LAYER_BLOCKED_RAYS,
        )
        if misses > 0
        else []
    )

    return rays_hit + rays_miss


def render_joint(dfk: HomMatrix) -> Any:
    dim, dtype = (
        dfk.shape[0] - 1,
        dfk.dtype,
    )

    origin = torch.zeros((dim,), dtype=dtype)
    joint = transform_points(dfk, origin)

    return [
        {
            "type": "points",
            "data": [joint.tolist()],
            "layers": [LAYER_JOINTS],
        }
    ]

