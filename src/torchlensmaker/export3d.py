import numpy as np
import torch
from os.path import join

import build123d as bd

import torchlensmaker as tlm

def tuplelist(arr):
    "Numpy array to list of tuples"

    return [tuple(e) for e in arr]


def sketch_line(line: tlm.Line):
    a, b, c = line.coefficients().detach().numpy().tolist()
    r = line.width / 2

    return bd.Polyline([
        (-r, (-c + a*r) / b),
        (r, (-c - a*r) / b),
    ])


def sketch_parabola(parabola: tlm.Parabola):
    a = parabola.coefficients().detach().numpy()
    r = parabola.width / 2

    return bd.Bezier([
        (-r, a*r**2),
        (0, -a*r**2),
        (r, a*r**2),
    ])


def sketch_circular_arc(arc: tlm.CircularArc):
    arc_radius = arc.coefficients().detach().item()
    x, y = arc.evaluate(arc.domain()[0]).detach().tolist()

    if arc_radius > 0:
        return bd.RadiusArc((-x, y), (x, y), arc_radius)
    else:
        return bd.RadiusArc((x, y), (-x, y), arc_radius)


def shape_to_sketch(shape):
    try:
        return {
            tlm.Parabola: sketch_parabola,
            tlm.Line: sketch_line,
            tlm.CircularArc: sketch_circular_arc,
        }[type(shape)](shape)
    except KeyError:
        raise RuntimeError(f"Unsupported shape type {type(shape)}")


def surface_to_sketch(surface):
    sketch = shape_to_sketch(surface.shape)
    scale = (*surface.scale.detach().tolist(), 1.)
    return bd.scale(sketch, scale)


def lens_to_part(lens):
    inner_thickness = lens.inner_thickness().detach().item()
    curve1 = surface_to_sketch(lens.surface1.surface)
    curve2 = bd.Pos(0., inner_thickness) * surface_to_sketch(lens.surface2.surface)

    # Find the "right most" point on the curve
    # i.e. extremity with the highest X value
    # This can be either sides depending on how the surface is parametrized
    v1 = curve1.vertices().sort_by(bd.Axis.X)[-1]
    v2 = curve2.vertices().sort_by(bd.Axis.X)[-1]

    # Connect them to form the lens edge
    edge = bd.Polyline([v1, v2])

    # Close the edges, revolve around the Y axis
    face = bd.make_face([curve1, edge, curve2])
    face = bd.split(face, bisect_by=bd.Plane.YZ)
    part = bd.revolve(face, bd.Axis.Y)

    return part


def export3d(optics, folder_path):
    "Export polygons of lenses in the the optical stack"

    for j, element in enumerate(optics):
        if isinstance(element, Lens):
            path = join(folder_path, f"lens{j}.step")
            part = lens_to_part(element)
            bd.export_step(part, path)
            yield part
