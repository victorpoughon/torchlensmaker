import torch
import torch.nn as nn
import torchlensmaker as tlm

def lens_thickness_gap(inner_thickness, outer_thickness):
    "Thickness and anchors for the provied thickness parametrization"

    if inner_thickness is not None and outer_thickness is None:
        thickness = inner_thickness
        anchors = ("origin", "origin")
    elif outer_thickness is not None and inner_thickness is None:
        thickness = outer_thickness
        anchors = ("origin", "extent")
    else:
        raise ValueError("Exactly one of inner/outer thickness must be given")

    return thickness, anchors


class GenericLens(tlm.Module):
    "A generic lens class providing common lens functions"

    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        return self.optics(inputs)

    def inner_thickness(self):
        "Thickness at the center of the lens"

        # Evaluate the lens stack with zero rays, just to forward compute positions
        execute_list, _ = tlm.full_forward(self, tlm.default_input)
        
        # Get the surfaces to compute thickness
        s1 = execute_list[0].module.surface(execute_list[0].inputs.target)
        s2 = execute_list[2].module.surface(execute_list[2].inputs.target)
        return torch.linalg.vector_norm(s1.at("origin") - s2.at("origin"))
    
    def outer_thickness(self):
        "Thickness at the outer radius of the lens"

        # Evaluate the lens stack with zero rays, just to forward compute positions
        execute_list, _ = tlm.full_forward(self, tlm.default_input)
        
        # Get the surfaces to compute thickness
        s1 = execute_list[0].module.surface(execute_list[0].inputs.target)
        s2 = execute_list[2].module.surface(execute_list[2].inputs.target)
        return torch.linalg.vector_norm(s1.at("extent") - s2.at("extent"))
    
    def thickness_at(self, x):
        "Thickness at distance x from the center of the lens"
        # TODO use collide to find intersection of both surfaces with X=x line
        return


class AsymmetricLens(GenericLens):
    """
    A lens made of two refractive surfaces with different shapes.
    """
    
    def __init__(self, shape1, shape2, n, inner_thickness=None, outer_thickness=None):
        super().__init__()
        self.shape1, self.shape2 = shape1, shape2

        thickness, anchors = lens_thickness_gap(inner_thickness, outer_thickness)

        self.surface1 = tlm.RefractiveSurface(self.shape1, n, anchors=anchors)
        self.gap = tlm.Gap(thickness)
        self.surface2 = tlm.RefractiveSurface(self.shape2, tuple(reversed(n)), anchors=tuple(reversed(anchors)))

        self.optics = nn.Sequential(self.surface1, self.gap, self.surface2)


class SymmetricLens(GenericLens):
    """
    A lens made of two symmetrical refractive surfaces of the same shape mirrored.
    """
    
    def __init__(self, shape, n, inner_thickness=None, outer_thickness=None):
        super().__init__()
        self.shape = shape

        thickness, anchors = lens_thickness_gap(inner_thickness, outer_thickness)

        self.surface1 = tlm.RefractiveSurface(self.shape, n, anchors=anchors)
        self.gap = tlm.Gap(thickness)
        self.surface2 = tlm.RefractiveSurface(self.shape, tuple(reversed(n)), scale=-1., anchors=tuple(reversed(anchors)))

        self.optics = nn.Sequential(self.surface1, self.gap, self.surface2)


class PlanoLens(GenericLens):
    """
    A plano-convex or plano-concave lens where one surface is curved
    as the given shape and the other surface is flat.

    By default the first surface is curved and the second is planar.
    This can be switched with the reverse argument:
    * reverse = False (default):  The curved side is the first surface
    * reverse = True:             The curved side is the second surface
    """

    # TODO make this an asymmetriclens?

    def __init__(self, shape, n, inner_thickness=None, outer_thickness=None, reverse=False):
        super().__init__()
        self.shape = shape

        thickness, anchors = lens_thickness_gap(inner_thickness, outer_thickness)
        line = tlm.Line(shape.height)

        if not reverse:
            self.surface1 = tlm.RefractiveSurface(self.shape, n, anchors=anchors)
            self.gap = tlm.Gap(thickness)
            self.surface2 = tlm.RefractiveSurface(line, tuple(reversed(n)))
        else:
            self.surface1 = tlm.RefractiveSurface(line, n)
            self.gap = tlm.Gap(thickness)
            self.surface2 = tlm.RefractiveSurface(self.shape, tuple(reversed(n)), scale=-1, anchors=tuple(reversed(anchors)))
        
        self.optics = nn.Sequential(self.surface1, self.gap, self.surface2)
        
