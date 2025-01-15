from IPython.display import display, HTML
import string
import uuid
import os.path
import json
import torch
import typing
Tensor = torch.Tensor

import torchlensmaker as tlm


def get_script_template() -> str:
    "Get the js script template string"

    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(dir_path, "script_template.js"), encoding="utf-8") as f:
        return "<script type='module'>" + f.read() + "</script>"


def get_div_template() -> str:
    return "<div data-jp-suppress-context-menu id='$div_id' style='width: 800px; height: 600px;'></div>"


def random_id() -> str:
    return f"tlmviewer-{uuid.uuid4().hex[:8]}"


def pprint(scene: object, ndigits: int | None = None):
    from pprint import pprint
    
    if ndigits is not None:
        json_data = json.dumps(scene, allow_nan=False)
        scene = json.loads(json_data, parse_float=lambda x: round(float(x), ndigits))
    
    pprint(scene)


def show(data: object, ndigits: int | None = None, dump: bool = False) -> None:
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


def render_surface(
    surface: tlm.surfaces.ImplicitSurface,
    transform: tlm.Surface2DTransform | tlm.Surface3DTransform,
    dim: int,
    N: int = 100,
) -> object:
    """
    Render a surface to a json serializable object in tlmviewer format
    """

    samples = surface.samples2D(N)

    if dim == 2 and isinstance(transform, tlm.Transform2DBase):
        front = torch.flip(
            torch.column_stack((samples[1:, 0], -samples[1:, 1])), dims=[0]
        )

        samples = torch.row_stack((front, samples))
        obj = {
            "matrix": transform.matrix3().tolist(),
            "samples": samples.tolist(),
        }
    elif dim == 3 and isinstance(transform, tlm.Transform3DBase):
        obj = {
            "matrix": transform.matrix4(surface).tolist(),
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


def render_rays(start: Tensor, end: Tensor, color: str = "#ffa724") -> dict[str, typing.Any]:
    data = torch.hstack((start, end)).tolist()
    return {
        "type": "rays",
        "data": data,
        "color": color,
    }


def render(
    rays_start: Tensor | None = None,
    rays_end: Tensor | None = None,
    points: Tensor | None = None,
    normals: Tensor | None = None,
    surfaces: list[tlm.surfaces.ImplicitSurface] | None = None,
    transforms: list[tlm.Surface3DTransform] | None = None,
    rays_color: str = "#ffa724",
) -> object:
    "Render tlm objects to json-able object"

    groups = []

    if surfaces is not None and transforms is not None:
        assert surfaces is not None
        groups.append(
            {
                "type": "surfaces",
                "data": [
                    render_surface(s, t, dim=3) for s, t in zip(surfaces, transforms)
                ],
            }
        )

    if rays_start is not None and rays_end is not None:
        groups.append(render_rays(rays_start, rays_end, rays_color))

    if points is not None:
        groups.append(
            {
                "type": "points",
                "data": points.tolist(),
                "color": "#ff0000",
            }
        )

    if normals is not None and points is not None:
        groups.append(
            {
                "type": "arrows",
                "data": [
                    n.tolist() + p.tolist() + [1.0] for p, n in zip(points, normals)
                ],
            }
        )

    return {"data": groups}
