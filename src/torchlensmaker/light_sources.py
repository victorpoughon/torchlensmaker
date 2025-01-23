import torch
import torch.nn as nn

from torchlensmaker.optics import OpticalData

from torchlensmaker.transforms import forward_kinematic
from torchlensmaker.rot2d import rot2d

from torchlensmaker.sampling import sample_line_linspace, sample_disk_linspace


def to_tensor(
    val: int | float | torch.Tensor,
    default_dtype=torch.float64,
) -> torch.Tensor:
    if not isinstance(val, torch.Tensor):
        return torch.as_tensor(val, dtype=default_dtype)
    return val


class PointSourceAtInfinity(nn.Module):
    def __init__(self, beam_diameter: float):
        """
        Args:
            beam_diameter: diameter of the beam of light
            angle_offset: incidence angle of the beam (in degrees)
        """
        super().__init__()
        self.beam_diameter = to_tensor(beam_diameter)

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


class PointSource(nn.Module):
    def __init__(self, beam_angular_size: float):
        super().__init__()

        self.beam_angular_size = torch.deg2rad(to_tensor(beam_angular_size))

    def forward(self, inputs: OpticalData) -> OpticalData:
        dim, dtype = inputs.sampling["dim"], inputs.sampling["dtype"]
        N = inputs.sampling["base"]

        if dim == 3:
            raise NotImplementedError

        # Sample coordinates other than X
        if dim == 2:
            theta = sample_line_linspace(N, self.beam_angular_size)
        else:
            # careful this does not sample exactly N points if N is not a perfect square
            theta = sample_disk_linspace(N, self.beam_angular_size)

        # Sample in cosine space
        unit = torch.tensor([1.0, 0.0], dtype=dtype)
        V = rot2d(unit, theta)

        P = torch.zeros_like(V)

        return inputs.replace(
            P=torch.cat((inputs.P, P), dim=0),
            V=torch.cat((inputs.V, V), dim=0),
        )
