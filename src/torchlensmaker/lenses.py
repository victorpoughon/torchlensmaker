import torch
import torch.nn as nn
import torchlensmaker as tlm


class SymmetricLens(tlm.Module):
    """
    A simple lens made of two symmetrical refractive surfaces.
    """
    
    def __init__(self, shape, n, inner_thickness=None, outer_thickness=None):
        super().__init__()
        self.shape = shape
        
        if inner_thickness is not None and outer_thickness is None:
            anchors = ("origin", "origin")
            thickness = inner_thickness
        elif outer_thickness is not None and inner_thickness is None:
            anchors = ("origin", "extent")
            thickness = outer_thickness
        else:
            raise ValueError("Exactly one of inner/outer thickness must be given")
        
        self.surface1 = tlm.RefractiveSurface(self.shape, n, anchors=anchors)
        self.gap = tlm.GapY(thickness)
        self.surface2 = tlm.RefractiveSurface(self.shape, tuple(reversed(n)), scale=-1., anchors=tuple(reversed(anchors))) 

        self.optics = nn.Sequential(self.surface1, self.gap, self.surface2)

    def forward(self, inputs):
        return self.optics(inputs)

    def inner_thickness(self):
        return torch.linalg.vector_norm(self.surface1.surface.at("origin") - self.surface2.surface.at("origin"))
    
    def outer_thickness(self):
        return torch.linalg.vector_norm(self.surface1.surface.at("extent") - self.surface2.surface.at("extent"))
