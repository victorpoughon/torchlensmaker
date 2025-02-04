from IPython.display import display, HTML
import string
import uuid
import os.path
import json
import torch

from typing import Any, Optional


import torchlensmaker as tlm

Tensor = torch.Tensor


def get_script_template() -> str:
    "Get the js script template string"

    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(dir_path, "script_template.js"), encoding="utf-8") as f:
        return "<script type='module'>" + f.read() + "</script>"


def get_div_template() -> str:
    return "<div data-jp-suppress-context-menu id='$div_id' style='width: 1000px; height: 650px;'></div>"


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


def ipython_display(
    data: object, ndigits: int | None = None, dump: bool = False
) -> None:
    div_id = random_id()
    div_template = get_div_template()
    script_template = get_script_template()

    json_data = json.dumps(data, allow_nan=False)

    if ndigits is not None:
        json_data = json.dumps(
            json.loads(json_data, parse_float=lambda x: round(float(x), ndigits))
        )

    if dump:
        print(json_data)

    div = string.Template(div_template).substitute(div_id=div_id)
    script = string.Template(script_template).substitute(data=json_data, div_id=div_id)
    display(HTML(div + script))  # type: ignore


def new_scene(mode: str) -> Any:
    if mode == "2D":
        return {"mode": "2D", "camera": "XY", "data": []}
    elif mode == "3D":
        return {"mode": "3D", "camera": "orthographic", "data": []}
    else:
        raise RuntimeError("mode should be 2D or 3D")


def process_surface(
    surface: tlm.surfaces.ImplicitSurface,
    transform: tlm.TransformBase,
    dim: int,
    N: int = 100,
) -> object:
    """
    Render a surface to a json serializable object in tlmviewer format
    """

    # TODO, should we have a type SymmetricSurface that provides samples2D?
    samples = surface.samples2D(N)

    if dim == 2:
        front = torch.flip(
            torch.column_stack((samples[1:, 0], -samples[1:, 1])), dims=[0]
        )

        samples = torch.row_stack((front, samples))
        obj = {
            "matrix": transform.hom_matrix().tolist(),
            "samples": samples.tolist(),
        }
    elif dim == 3:
        obj = {
            "matrix": transform.hom_matrix().tolist(),
            "samples": samples.tolist(),
        }
    else:
        raise RuntimeError("inconsistent arguments to render_surface")

    # convert the outline to clip planes
    if dim == 3:
        clip_planes = surface.outline.clip_planes()
        if len(clip_planes) > 0:
            obj["clip_planes"] = clip_planes

    return obj


def render_surfaces(
    surfaces: list[tlm.LocalSurface],
    transforms: list[tlm.TransformBase],
    dim: int,
    N: int = 100,
) -> Any:
    return {
        "type": "surfaces",
        "data": [process_surface(s, t, dim, N) for s, t in zip(surfaces, transforms)],
    }


def render_rays(
    start: Tensor,
    end: Tensor,
    variables: dict[str, Tensor],
    domain: dict[str, list[float]],
    default_color: str = "#ffa724",
    layer: Optional[int] = None,
) -> Any:

    assert start.shape == end.shape
    for var in variables.values():
        assert var.shape[0] == start.shape[0]

    var_tensors = tuple(
        var.unsqueeze(1) for var in variables.values()
    )

    data = torch.hstack((start, end) + var_tensors).tolist()

    node = {
        "type": "rays",
        "data": data,
        "color": default_color,
        "variables": list(variables.keys()),
        "domain": domain,
    }

    if layer is not None:
        node.update(layers=[layer])

    return node

def render_points(points: Tensor, color: str = "white") -> Any:
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
