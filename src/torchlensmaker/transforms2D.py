import torch
from torchlensmaker.surfaces import LocalSurface
import functools

# for shorter type annotations
Tensor = torch.Tensor


def homogeneous_transform_matrix3(A: Tensor, B: Tensor) -> Tensor:
    "Homogeneous 3x3 transform matrix for 2D transform AX+B"
    rows = torch.cat((A, B.unsqueeze(0).T), dim=1)
    return torch.cat((rows, torch.tensor([[0.0, 0.0, 1.0]])), dim=0)


class Transform2DBase:
    "Abstract base class for 2D transforms"

    def direct_points(self, points: Tensor) -> Tensor:
        raise NotImplementedError

    def direct_vectors(self, vectors: Tensor) -> Tensor:
        raise NotImplementedError

    def inverse_points(self, points: Tensor) -> Tensor:
        raise NotImplementedError

    def inverse_vectors(self, vectors: Tensor) -> Tensor:
        raise NotImplementedError

    def matrix3(self) -> Tensor:
        raise NotImplementedError


# testing transforms
# compose inverse with direct == identity for both vector and points
#


class Translate2D(Transform2DBase):
    "2D translation: Y = X + T"

    def __init__(self, T: Tensor):
        self.T = T

    def direct_points(self, points: Tensor) -> Tensor:
        return points + self.T

    def direct_vectors(self, vectors: Tensor) -> Tensor:
        return vectors

    def inverse_points(self, points: Tensor) -> Tensor:
        return points - self.T

    def inverse_vectors(self, vectors: Tensor) -> Tensor:
        return vectors

    def matrix3(self) -> Tensor:
        return torch.row_stack(
            (
                torch.column_stack(
                    (torch.eye(2, dtype=self.T.dtype), self.T.unsqueeze(0).T)
                ),
                torch.tensor([0, 0, 1], dtype=self.T.dtype),
            )
        )


class Linear2D(Transform2DBase):
    "Linear 2D transform: Y = AX"

    def __init__(self, A: Tensor, A_inv: Tensor):
        self.A = A
        self.A_inv = A_inv

    def direct_points(self, points: Tensor) -> Tensor:
        return (self.A @ points.T).T

    def direct_vectors(self, vectors: Tensor) -> Tensor:
        return (self.A @ vectors.T).T

    def inverse_points(self, points: Tensor) -> Tensor:
        return (self.A_inv @ points.T).T

    def inverse_vectors(self, vectors: Tensor) -> Tensor:
        return (self.A_inv @ vectors.T).T

    def matrix3(self) -> Tensor:
        return torch.row_stack(
            (
                torch.column_stack((self.A, torch.zeros((2, 1), dtype=self.A.dtype))),
                torch.tensor([0, 0, 1], dtype=self.A.dtype),
            )
        )


class SurfaceExtent2D(Transform2DBase):
    "Translation from a surface extent point"

    def __init__(self, surface: LocalSurface):
        self.surface = surface

    def _extent(self) -> Tensor:
        return torch.cat((self.surface.extent().unsqueeze(0), torch.zeros(1)), dim=0)

    def direct_points(self, points: Tensor) -> Tensor:
        return points - self._extent()

    def direct_vectors(self, vectors: Tensor) -> Tensor:
        return vectors

    def inverse_points(self, points: Tensor) -> Tensor:
        return points + self._extent()

    def inverse_vectors(self, vectors: Tensor) -> Tensor:
        return vectors

    def matrix3(self) -> Tensor:
        return torch.row_stack(
            (
                torch.column_stack(
                    (
                        torch.eye(2, dtype=self.surface.dtype),
                        -self._extent().unsqueeze(0).T,
                    )
                ),
                torch.tensor([0, 0, 1], dtype=self.surface.dtype),
            )
        )


class Compose2D(Transform2DBase):
    "Compose two 2D transforms"

    def __init__(self, T1: Transform2DBase, T2: Transform2DBase):
        self.T1 = T1
        self.T2 = T2

    def direct_points(self, points: Tensor) -> Tensor:
        return self.T2.direct_points(self.T1.direct_points(points))

    def direct_vectors(self, vectors: Tensor) -> Tensor:
        return self.T2.direct_vectors(self.T1.direct_vectors(vectors))

    def inverse_points(self, points: Tensor) -> Tensor:
        return self.T1.inverse_points(self.T2.inverse_points(points))

    def inverse_vectors(self, vectors: Tensor) -> Tensor:
        return self.T1.inverse_vectors(self.T2.inverse_vectors(vectors))

    def matrix3(self) -> Tensor:
        return self.T2.matrix3() @ self.T1.matrix3()


class ComposeList2D(Transform2DBase):
    "Compose a list of 2D transforms"

    def __init__(self, transforms: list[Transform2DBase]):
        self.transforms = transforms

    def direct_points(self, points: Tensor) -> Tensor:
        for t in self.transforms:
            points = t.direct_points(points)
        return points

    def direct_vectors(self, vectors: Tensor) -> Tensor:
        for t in self.transforms:
            vectors = t.direct_vectors(vectors)
        return vectors

    def inverse_points(self, points: Tensor) -> Tensor:
        for t in reversed(self.transforms):
            points = t.inverse_points(points)
        return points

    def inverse_vectors(self, vectors: Tensor) -> Tensor:
        for t in reversed(self.transforms):
            vectors = t.inverse_vectors(vectors)
        return vectors

    def matrix3(self) -> Tensor:
        return functools.reduce(
            lambda t1, t2: t2 @ t1, [t.matrix3() for t in self.transforms]
        )


class Surface2DTransform:
    """
    2D Transform of the form X' = RS(X - A) + T
    where A is a surface anchor point determined by the surface shape
    """

    def __init__(
        self, scale: float, anchor: str, rotation: float, position: list[float]
    ):
        self.anchor = anchor

        # scale matrix
        self.S = torch.tensor([[scale, 0.0], [0.0, 1.0]])
        self.S_inv = torch.tensor([[1.0 / scale, 0.0], [0.0, 1.0]])

        # rotation matrix
        theta = torch.deg2rad(torch.as_tensor(rotation))
        self.R = torch.tensor(
            [
                [torch.cos(theta), -torch.sin(theta)],
                [torch.sin(theta), torch.cos(theta)],
            ]
        )

        self.R_inv = self.R.T

        # position translation
        self.T = torch.as_tensor(position)

    def anchor_point(self, surface: LocalSurface) -> Tensor:
        "Get position of anchor of surface"
        if self.anchor == "origin":
            return torch.zeros(2)
        elif self.anchor == "extent":
            return torch.cat((surface.extent().unsqueeze(0), torch.zeros(1)), dim=0)
        else:
            raise ValueError("anchor must be one of origin/extent")

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

    def matrix3(self, surface: LocalSurface) -> Tensor:

        hom = homogeneous_transform_matrix3
        return hom(self.R @ self.S, self.T) @ hom(
            torch.eye(2), -self.anchor_point(surface)
        )
