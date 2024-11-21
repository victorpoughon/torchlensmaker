import numpy as np
import torch
from os.path import join

import build123d as bd

from .optics import Lens

from torchlensmaker.shapes import (
    CircularArc,
    Line,
    Parabola,
)

def tuplelist(arr):
    "Numpy array to list of tuples"

    return [tuple(e) for e in arr]


def sketch_line(line, offset):
    a, b, c = line.coefficients().detach().numpy().tolist()
    r = line.width

    return bd.Polyline([
        (-r, (-c + a*r) / b + offset),
        (r, (-c - a*r) / b + offset),
    ])


def sketch_parabola(parabola, offset):
    a = parabola.coefficients().detach().numpy()
    r = parabola.width

    return bd.Bezier([
        (-r, a*r**2 + offset),
        (0, -a*r**2 + offset),
        (r, a*r**2 + offset),
    ])


def sketch_circular_arc(arc: CircularArc, offset):
    arc_radius = arc.coefficients().detach().item()
    x = arc.width
    y = arc.evaluate(arc.domain()[0]).detach()[1]

    return bd.RadiusArc((-x, y + offset), (x, y + offset), arc_radius)


def shape_to_sketch(surface, offset):
    # Dynamic dispatch for dummies
    try:
        return {
            Parabola: sketch_parabola,
            Line: sketch_line,
            CircularArc: sketch_circular_arc,
        }[type(surface)](surface, offset)
    except KeyError:
        raise RuntimeError(f"Unsupported shape type {type(surface)}")


def lens_to_part(surface1, gap, surface2):

    gap_size = lens.thickness()[0].item()
    curve1 = shape_to_sketch(surface1.surface, 0.0)
    curve2 = shape_to_sketch(surface2.surface, gap_size)

    edge = bd.Polyline([
        curve1 @ 1,
        curve2 @ 1,
    ])

    # Close the edges
    face = bd.make_face([curve1, edge, curve2])
    face = bd.split(face, bisect_by=bd.Plane.YZ)

    part = bd.revolve(face, bd.Axis.Y)

    return part
    

def export_lens(lens: Lens, folder_path):
    surface1 = lens[0].surface.XY()
    gap_size = lens[1].origin
    surface2 = lens[2].surface.XY()

    print(surface1)
    print(gap_size)
    print(surface2)
    # profile = element.profile
    # # Interleave A and B to make polygon
    # X, Y = element.profile.XY()
    # data = np.column_stack((X, Y))
    # filepath = join(folder_path, f"lens{str(j)}.npy")
    # np.save(filepath, data)


def export3d(optics, folder_path):
    "Export polygons of lenses in the the optical stack"

    for j, element in enumerate(optics):
        if isinstance(element, Lens):
            path = join(folder_path, f"lens{j}.step")
            part = lens_to_part(element)
            bd.export_step(part, path)
            yield part
    