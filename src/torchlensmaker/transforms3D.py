import torch

from torchlensmaker.surfaces import LocalSurface

from torchlensmaker.rot3d import euler_angles_to_matrix

Tensor = torch.Tensor


def homogeneous_transform_matrix4(A: Tensor, B: Tensor) -> Tensor:
    "Homogeneous 4x4 transform matrix for 3D transform AX+B"
    rows = torch.cat((A, B.unsqueeze(0).T), dim=1)
    return torch.cat((rows, torch.tensor([[0.0, 0.0, 0.0, 1.0]])), dim=0)


class Surface3DTransform:
    """
    3D transform of the form X' = RS(X - A) + T
    where A is a surface anchor point determined by the surface shape
    """

    def __init__(
        self, scale: float, anchor: str, rotations: list[float], position: list[float]
    ):
        self.anchor = anchor

        # scale matrix
        self.S = torch.tensor([[scale, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        self.S_inv = torch.tensor(
            [[1.0 / scale, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )

        # rotation matrix
        self.R = euler_angles_to_matrix(
            torch.deg2rad(torch.as_tensor(rotations)), "XYZ"
        )
        self.R_inv = self.R.T

        # position translation
        self.T = torch.as_tensor(position)

    def anchor_point(self, surface: LocalSurface) -> Tensor:
        "Get position of anchor of surface"
        if self.anchor == "origin":
            return torch.zeros(3)
        elif self.anchor == "extent":
            return torch.cat((surface.extent().unsqueeze(0), torch.zeros(2)), dim=0)
        else:
            raise ValueError

    def direct_vectors(self, V: Tensor) -> Tensor:
        return (self.R @ self.S @ V.T).T

    def inverse_points(self, surface: LocalSurface, P: Tensor) -> Tensor:
        S_inv, R_inv, T = self.S_inv, self.R_inv, self.T
        A = self.anchor_point(surface)
        return (S_inv @ R_inv @ (P - T).T).T + A

    def inverse_rays(
        self, P: Tensor, V: Tensor, surface: LocalSurface
    ) -> tuple[Tensor, Tensor]:
        S_inv, R_inv, T = self.S_inv, self.R_inv, self.T
        A = self.anchor_point(surface)

        Ps = (S_inv @ R_inv @ (P - T).T).T + A
        Vs = (S_inv @ R_inv @ V.T).T

        return Ps, Vs

    def matrix4(self, surface: LocalSurface) -> Tensor:
        hom = homogeneous_transform_matrix4
        return hom(self.R @ self.S, self.T) @ hom(
            torch.eye(3), -self.anchor_point(surface)
        )



