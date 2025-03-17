from IPython.display import display, HTML
import string
import uuid
import os.path
import json
import torch

import warnings
from pathlib import Path

from importlib.metadata import version, PackageNotFoundError

from typing import Any, Optional

from torchlensmaker.core.surfaces import (
    LocalSurface,
    ImplicitSurface,
    Plane,
)

from torchlensmaker.core.transforms import (
    TransformBase,
    IdentityTransform,
)

Tensor = torch.Tensor


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


def process_surface(
    surface: ImplicitSurface,
    transform: TransformBase,
    dim: int,
    N: int = 100,
) -> object:
    """
    Render a surface to a json serializable object in tlmviewer format
    """

    # TODO, should we have a type SymmetricSurface that provides samples2D?

    if dim == 2:
        samples = surface.samples2D_full(N, epsilon=0.)
        obj = {
            "matrix": transform.hom_matrix().tolist(),
            "samples": samples.tolist(),
        }
    elif dim == 3:
        samples = surface.samples2D_half(N, epsilon=0.)
        obj = {
            "matrix": transform.hom_matrix().tolist(),
            "samples": samples.tolist(),
        }
    else:
        raise RuntimeError("inconsistent arguments to render_surface")

    # convert the outline to clip planes
    if dim == 3 and isinstance(surface, Plane):
        clip_planes = surface.outline.clip_planes()
        if len(clip_planes) > 0:
            obj["clip_planes"] = clip_planes

    return obj


def render_surfaces(
    surfaces: list[LocalSurface],
    transforms: list[TransformBase],
    dim: int,
    N: int = 100,
) -> Any:
    return {
        "type": "surfaces",
        "data": [process_surface(s, t, dim, N) for s, t in zip(surfaces, transforms)],
    }

def render_surface(
        surface: LocalSurface,
        dim: int,
) -> Any:
    return render_surfaces([surface], [IdentityTransform(dim=dim, dtype=surface.dtype)], dim=dim)

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

def render_points(points: Tensor, color: str = "white") -> Any:
    # TODO render points sizes in screen coordinates
    assert points.dim() == 2
    return {
        "type": "points",
        "data": points.tolist(),
        "color": color,
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
