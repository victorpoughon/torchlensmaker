import torch

from torchlensmaker.surfaces import LocalSurface3D

from torchlensmaker.rot3d import euler_angles_to_matrix

Tensor = torch.Tensor


def homogeneous_transform_matrix4(A: Tensor, B: Tensor) -> Tensor:
    "Homogeneous 4x4 transform matrix for 3D transform AX+B"
    rows = torch.cat((A, B.unsqueeze(0).T), dim=1)
    return torch.cat((rows, torch.tensor([[0.0, 0.0, 0.0, 1.0]])), dim=0)


class BaseTransform:
    def direct_vectors(self, vectors: Tensor) -> Tensor:
        "Apply the transform to vectors"
        raise NotImplementedError

    def inverse_points(self, surface: LocalSurface3D, points: Tensor) -> Tensor:
        "Apply the inverse transform to points"
        raise NotImplementedError

    def inverse_rays(
        self, P: Tensor, V: Tensor, surface: LocalSurface3D
    ) -> tuple[Tensor, Tensor]:
        "Apply the inverse transform to rays"
        raise NotImplementedError

    def matrix4(self, surface: LocalSurface3D) -> Tensor:
        "Homogeneous coordinates 4x4 matrix representing the transform"
        raise NotImplementedError


class SurfaceTransform(BaseTransform):
    """
    Transform of the form X' = RS(X - A) + T
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

    def anchor_point(self, surface: LocalSurface3D) -> Tensor:
        "Get position of anchor of surface"
        if self.anchor == "origin":
            return torch.zeros(3)
        elif self.anchor == "extent":
            return torch.cat((surface.extent().unsqueeze(0), torch.zeros(2)), dim=0)
        else:
            raise ValueError

    def direct_vectors(self, V: Tensor) -> Tensor:
        return (self.R @ self.S @ V.T).T

    def inverse_points(self, surface: LocalSurface3D, P: Tensor) -> Tensor:
        S_inv, R_inv, T = self.S_inv, self.R_inv, self.T
        A = self.anchor_point(surface)
        return (S_inv @ R_inv @ (P - T).T).T + A

    def inverse_rays(
        self, P: Tensor, V: Tensor, surface: LocalSurface3D
    ) -> tuple[Tensor, Tensor]:
        S_inv, R_inv, T = self.S_inv, self.R_inv, self.T
        A = self.anchor_point(surface)

        Ps = (S_inv @ R_inv @ (P - T).T).T + A
        Vs = (S_inv @ R_inv @ V.T).T

        return Ps, Vs

    def matrix4(self, surface: LocalSurface3D) -> Tensor:
        hom = homogeneous_transform_matrix4
        return hom(self.R @ self.S, self.T) @ hom(
            torch.eye(3), -self.anchor_point(surface)
        )


def intersect(
    surface: LocalSurface3D, P: Tensor, V: Tensor, transform: BaseTransform
) -> tuple[Tensor, Tensor]:
    """
    Surface-rays collision detection

    Find collision points and normal vectors for the intersection of rays P+tV with
    a surface and a transform applied to that surface.

    Args:
        P: (N,3) tensor, rays origins
        V: (N, 3) tensor, rays vectors
        surface: surface to collide with
        transform: transform applied to the surface

    Returns:
        points: collision points
        normals: surface normals at the collision points
    """

    # Convert rays to surface local frame
    Ps, Vs = transform.inverse_rays(P, V, surface)

    # Collision detection in the surface local frame
    t, local_normals, valid = surface.local_collide(Ps, Vs)

    # Compute collision points and convert normals to global frame
    points = P + t.unsqueeze(1).expand((-1, 3)) * V
    normals = transform.direct_vectors(local_normals)

    # remove non valid (non intersecting) points
    # do this before computing global frame?
    points = points[valid]
    normals = normals[valid]

    return points, normals