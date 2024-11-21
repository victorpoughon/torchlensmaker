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
            anchors = ("origin", "extent")
            thickness = inner_thickness 
        elif outer_thickness is not None and inner_thickness is None:
            anchors = ("origin", "origin")
            thickness = outer_thickness
        else:
            raise ValueError("Exactly one of inner/outer thickness must be given")

        self.optics = nn.Sequential(
            tlm.RefractiveSurface(self.shape, n, anchors=anchors),
            tlm.GapY(thickness),
            tlm.RefractiveSurface(self.shape, tuple(reversed(n)), scale=-1., anchors=tuple(reversed(anchors))),  
        )

    def forward(self, inputs):
        return self.optics(inputs)
