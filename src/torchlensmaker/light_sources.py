import torch
import torch.nn as nn

from torchlensmaker.optics import OpticalData

from torchlensmaker.transforms import forward_kinematic
from torchlensmaker.rot2d import rot2d
from torchlensmaker.rot3d import euler_angles_to_matrix

from torchlensmaker.sampling import SampleDisk


Tensor = torch.Tensor


def to_tensor(
    val: int | float | torch.Tensor,
    default_dtype: torch.dtype = torch.float64,
) -> Tensor:
    if not isinstance(val, torch.Tensor):
        return torch.as_tensor(val, dtype=default_dtype)
    return val


def unit_vector(dim: int, dtype: torch.dtype) -> Tensor:
    return torch.cat((torch.ones(1, dtype=dtype), torch.zeros(dim - 1, dtype=dtype)))


def rotated_unit_vector(angles: Tensor, dim: int) -> Tensor:
    """
    Rotated unit X vector in 2D or 3D
    angles is batched with shape (N, 2|3)
    """

    dtype = angles.dtype
    N = angles.shape[0]
    if dim == 2:
        unit = torch.tensor([1.0, 0.0], dtype=dtype)
        return rot2d(unit, angles)
    else:
        unit = torch.tensor([1.0, 0.0, 0.0], dtype=dtype)
        thetas = torch.column_stack(
            (
                torch.zeros(N, dtype=dtype),
                angles,
            )
        )
        M = euler_angles_to_matrix(thetas, "XZY").to(
            dtype=dtype
        )  # TODO need to support dtype in euler_angles_to_matrix
        return torch.matmul(M, unit.view(3, 1)).squeeze(-1)


class PointSourceAtInfinity(nn.Module):
    def __init__(self, beam_diameter: float):
        """
        Args:
            beam_diameter: diameter of the beam of light
            angle_offset: incidence angle of the beam (in degrees)
        """
        super().__init__()
        self.beam_diameter: Tensor = to_tensor(beam_diameter)

    def forward(self, inputs: OpticalData) -> OpticalData:
        dim, dtype = inputs.sampling["dim"], inputs.sampling["dtype"]
        N = inputs.sampling["base"]

        # Sample coordinates other than X on a disk
        NX = SampleDisk.sample(N, self.beam_diameter, dim)

        # Make the rays P + tV
        P = torch.column_stack((torch.zeros(NX.shape[0], dtype=dtype), NX))
        V = torch.tile(unit_vector(dim, dtype), (P.shape[0], 1))

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

        # Sample angular direction
        angles = SampleDisk.sample(N, self.beam_angular_size, dim)

        V = rotated_unit_vector(angles, dim)
        P = torch.zeros_like(V)

        # Apply kinematic transform
        tf = forward_kinematic(inputs.transforms)
        P = tf.direct_points(P)
        V = tf.direct_vectors(V)

        return inputs.replace(
            P=torch.cat((inputs.P, P), dim=0),
            V=torch.cat((inputs.V, V), dim=0),
        )
