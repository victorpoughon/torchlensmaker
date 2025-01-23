import torch
import torch.nn as nn

from torchlensmaker.optics import (
    OpticalData
)

from torchlensmaker.transforms import forward_kinematic

from torchlensmaker.sampling import sample_line_linspace, sample_disk_linspace


class PointSourceAtInfinity(nn.Module):
    def __init__(self, beam_diameter: float):
        """
        Args:
            beam_diameter: diameter of the beam of light
            angle_offset: incidence angle of the beam (in degrees)
        """
        super().__init__()
        self.beam_diameter = beam_diameter

    def forward(self, inputs: OpticalData) -> OpticalData:
        dim, dtype = inputs.sampling["dim"], inputs.sampling["dtype"]

        N = inputs.sampling["base"]

        # Sample coordinates other than X
        if dim == 2:
            NX = sample_line_linspace(N, self.beam_diameter)
        else:
            # careful this does not sample exactly N points if N is not a perfect square
            NX = sample_disk_linspace(N, self.beam_diameter)

        # Make the rays P + tV
        P = torch.column_stack((torch.zeros(NX.shape[0], dtype=dtype), NX))
        unit_vect = torch.cat(
            (torch.ones(1, dtype=dtype), torch.zeros(dim - 1, dtype=dtype))
        )
        V = torch.tile(unit_vect, (P.shape[0], 1))

        # Make the rays 'base' coordinate
        base = NX

        # Apply kinematic transform
        tf = forward_kinematic(inputs.transforms)
        P = tf.direct_points(P)
        V = tf.direct_vectors(V)

        # Concatenate with existing rays
        # TODO some check that there are no other variables?
        return inputs.replace(
            P=torch.cat((inputs.P, P), dim=0),
            V=torch.cat((inputs.V, V), dim=0),
        )
