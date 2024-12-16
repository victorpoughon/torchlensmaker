import numpy as np
import torch
from os.path import join

import build123d as bd

import torchlensmaker as tlm

def tuplelist(arr):
    "Numpy array to list of tuples"

    return [tuple(e) for e in arr]


def sketch_line(line: tlm.Line):
    y = line.height / 2
    return bd.Line([(0, -y), (0, y)])


def sketch_parabola(parabola: tlm.Parabola):
    a = parabola.coefficients().detach().numpy()
    r = parabola.height / 2

    return bd.Bezier([
        (a*r**2, -r),
        (-a*r**2, 0),
        (a*r**2, r),
    ])


def sketch_circular_arc(arc: tlm.CircularArc):
    arc_radius = arc.coefficients().detach().item()
    x, y = arc.evaluate(arc.domain()[0]).detach().tolist()

    print(x, y, arc_radius)

    if arc_radius > 0:
        return bd.RadiusArc((x, y), (x, -y), arc_radius)
    else:
        return bd.RadiusArc((x, -y), (x, y), arc_radius)


def shape_to_sketch(shape):
    try:
        return {
            tlm.Parabola: sketch_parabola,
            tlm.Line: sketch_line,
            tlm.CircularArc: sketch_circular_arc,
        }[type(shape)](shape)
    except KeyError:
        raise RuntimeError(f"Unsupported shape type {type(shape)}")


def lens_to_part(lens):
    inner_thickness = lens.inner_thickness().detach().item()
    
    curve1 = bd.scale(shape_to_sketch(lens.surface1.shape), (lens.surface1.scale, 1., 1.))
    curve2 = bd.Pos(inner_thickness, 0.) * bd.scale(shape_to_sketch(lens.surface2.shape), (lens.surface2.scale, 1., 1.))

    # Find the "top most" point on the curve
    # i.e. extremity with the highest Y value
    # This can be either sides depending on how the surface is parametrized
    v1 = curve1.vertices().sort_by(bd.Axis.Y)[-1]
    v2 = curve2.vertices().sort_by(bd.Axis.Y)[-1]

    # Connect them to form the lens edge
    edge = bd.Polyline([v1, v2])

    # Close the edges
    face = bd.make_face([curve1, edge, curve2])

    # split only the side we are going to revolve
    # keep bottom side because the normal direction of Plane.XZ is in the
    # negative y-direction according to the right-hand-rule
    face = bd.split(face, bisect_by=bd.Plane.XZ, keep=bd.Keep.BOTTOM)

    # revolve around the X axis
    part = bd.revolve(face, bd.Axis.X)

    return part


def export3d(optics, folder_path):
    "Export polygons of lenses in the the optical stack"

    for j, element in enumerate(optics):
        if isinstance(element, Lens):
            path = join(folder_path, f"lens{j}.step")
            part = lens_to_part(element)
            bd.export_step(part, path)
            yield part
