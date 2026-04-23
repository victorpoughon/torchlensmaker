from dataclasses import dataclass, field
from typing import Any, Literal

Matrix = list[list[float]]
ClipPlane = tuple[float, float, float, float]


@dataclass
class AmbientLight:
    color: str
    intensity: float


@dataclass
class DirectionalLight:
    color: str
    intensity: float
    position: tuple[float, float, float]


@dataclass
class SceneAxis:
    axis: Literal["x", "y", "z"]
    length: float
    color: str


@dataclass
class SceneTitle:
    title: str


@dataclass
class Arrows:
    arrows: list[list[float]]


@dataclass
class Points:
    data: list[list[float]]
    color: str
    radius: float
    category: str


@dataclass
class Rays:
    points: list[list[float]]
    color: str
    category: str
    dim: Literal[2, 3] = 3
    variables: dict[str, list[float]] = field(default_factory=dict)
    domain: dict[str, tuple[float, float]] = field(default_factory=dict)


@dataclass
class Box3D:
    size: tuple[float, float, float]
    matrix: Matrix


@dataclass
class Cylinder:
    xmin: float
    xmax: float
    radius: float
    matrix: Matrix


@dataclass
class SurfaceDisk:
    radius: float
    matrix: Matrix
    clip_planes: list[ClipPlane] = field(default_factory=list)


@dataclass
class SurfaceLathe:
    samples: list[list[float]]
    matrix: Matrix
    clip_planes: list[ClipPlane] = field(default_factory=list)


@dataclass
class SurfaceSphereR:
    R: float
    diameter: float
    matrix: Matrix
    clip_planes: list[ClipPlane] = field(default_factory=list)


@dataclass
class SurfaceSag:
    diameter: float
    sag_function: dict[str, Any]
    matrix: Matrix
    clip_planes: list[ClipPlane] = field(default_factory=list)


@dataclass
class SurfaceBSpline:
    points: list[list[list[float]]]
    weights: list[list[float]]
    degree: tuple[int, int]
    knot_type: Literal["clamped", "unclamped"]
    samples: tuple[int, int]
    matrix: Matrix
    clip_planes: list[ClipPlane] = field(default_factory=list)


SceneElement = (
    AmbientLight
    | DirectionalLight
    | SceneAxis
    | SceneTitle
    | Arrows
    | Points
    | Rays
    | Box3D
    | Cylinder
    | SurfaceDisk
    | SurfaceLathe
    | SurfaceSphereR
    | SurfaceSag
    | SurfaceBSpline
)


@dataclass
class Scene:
    data: list[SceneElement] = field(default_factory=list)
    mode: Literal["3D", "2D"] = "3D"
    camera: str | None = None
    controls: dict[str, Any] = field(default_factory=dict)
