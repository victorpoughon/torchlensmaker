import dataclasses
import json
from typing import Any

from .types import (
    Scene,
    AmbientLight,
    DirectionalLight,
    SceneAxis,
    SceneTitle,
    Arrows,
    Points,
    Rays,
    Box3D,
    Cylinder,
    SurfaceDisk,
    SurfaceLathe,
    SurfaceSphere,
    SurfaceSphereR,
    SurfaceSag,
    SurfaceBSpline,
)

_ELEMENT_TYPE_STRINGS: dict[type, str] = {
    AmbientLight: "ambient-light",
    DirectionalLight: "directional-light",
    SceneAxis: "scene-axis",
    SceneTitle: "scene-title",
    Arrows: "arrows",
    Points: "points",
    Rays: "rays",
    Box3D: "box3D",
    Cylinder: "cylinder",
    SurfaceDisk: "surface-disk",
    SurfaceLathe: "surface-lathe",
    SurfaceSphere: "surface-sphere",
    SurfaceSphereR: "surface-sphere-r",
    SurfaceSag: "surface-sag",
    SurfaceBSpline: "surface-bspline",
}

_FIELD_RENAMES: dict[str, str] = {
    "clip_planes": "clipPlanes",
    "knot_type": "knotType",
    "sag_function": "sag-function",
}


def _element_to_dict(element: Any) -> dict[str, Any]:
    d = dataclasses.asdict(element)
    for old, new in _FIELD_RENAMES.items():
        if old in d:
            d[new] = d.pop(old)
    d["type"] = _ELEMENT_TYPE_STRINGS[type(element)]
    return d


def scene_to_dict(scene: Scene) -> dict[str, Any]:
    d: dict[str, Any] = {
        "mode": scene.mode,
        "data": [_element_to_dict(e) for e in scene.data],
    }
    if scene.camera is not None:
        d["camera"] = scene.camera
    if scene.controls:
        d["controls"] = scene.controls
    return d


def scene_to_json(scene: Scene) -> str:
    return json.dumps(scene_to_dict(scene))


def save_scene(scene: Scene, path: str) -> None:
    with open(path, "w") as f:
        json.dump(scene_to_dict(scene), f)
