import torch
import torch.nn as nn
import torchlensmaker as tlm
import build123d as bd
from os.path import join

from typing import Any


def tuplelist(arr: Any) -> Any:
    "Numpy array to list of tuples"

    return [tuple(e) for e in arr]


def sketch_circular_plane(plane: tlm.CircularPlane) -> bd.Sketch:
    y = plane.outline.max_radius()
    return bd.Line([(0, -y), (0, y)])


def sketch_parabola(parabola: tlm.Parabola) -> bd.Sketch:
    a: float = parabola.A.item()
    r: float = parabola.diameter / 2

    return bd.Bezier((a * r**2, -r), (-a * r**2, 0), (a * r**2, r))


def sketch_sphere(sphere: tlm.Sphere) -> bd.Sketch:
    y = -sphere.diameter / 2
    X = sphere.extent_x().item()
    R = sphere.radius()

    if R > 0:
        return bd.RadiusArc((X, y), (X, -y), R)
    else:
        return bd.RadiusArc((X, -y), (X, y), -R)


def surface_to_sketch(surface: tlm.LocalSurface) -> bd.Sketch:
    try:
        func = {
            tlm.Parabola: sketch_parabola,
            tlm.CircularPlane: sketch_circular_plane,
            tlm.Sphere: sketch_sphere,
        }[type(surface)]
        return func(surface)  # type: ignore
    except KeyError:
        raise RuntimeError(f"Unsupported surface type {type(surface)}")


def lens_to_part(lens: tlm.LensBase) -> bd.Part:
    inner_thickness = lens.inner_thickness().detach().item()

    curve1 = bd.scale(
        surface_to_sketch(lens.surface1.surface), (lens.surface1.scale, 1.0, 1.0)
    )
    curve2 = bd.Pos(inner_thickness, 0.0) * bd.scale(
        surface_to_sketch(lens.surface2.surface), (lens.surface2.scale, 1.0, 1.0)
    )

    # Find the "top most" point on the curve
    # i.e. extremity with the highest Y value
    # This can be either sides depending on how the surface is parametrized
    v1 = curve1.vertices().sort_by(bd.Axis.Y)[-1]
    v2 = curve2.vertices().sort_by(bd.Axis.Y)[-1]

    # Connect them to form the lens edge
    if (bd.Vector(v1) - bd.Vector(v2)).length > 1e-6:
        edge = bd.Polyline([v1, v2])
        face = bd.make_face([curve1, edge, curve2])
    else:
        face = bd.make_face([curve1, curve2])

    # split only the side we are going to revolve
    # keep bottom side because the normal direction of Plane.XZ is in the
    # negative y-direction according to the right-hand-rule
    face = bd.split(face, bisect_by=bd.Plane.XZ, keep=bd.Keep.BOTTOM)

    # revolve around the X axis
    part = bd.revolve(face, bd.Axis.X)

    return part


def export_all_step(optics: nn.Sequential, folder_path: str) -> None:
    "Export polygons of lenses in the the optical stack"

    for j, element in enumerate(optics):
        if isinstance(element, tlm.LensBase):
            path = join(folder_path, f"lens{j}.step")
            part = lens_to_part(element)
            bd.export_step(part, path)
