import torch
from torchlensmaker.surfaces import LocalSurface
import functools

from typing import Iterable

# for shorter type annotations
Tensor = torch.Tensor


def hom_matrix(A: Tensor, B: Tensor) -> Tensor:
    "Homogeneous transform matrix for transform AX+B"
    assert A.dim() == 2
    assert B.dim() == 1
    assert A.shape[0] == A.shape[1] == B.shape[0]
    assert A.dtype == B.dtype, (A.dtype, B.dtype)
    dim = A.shape[0]
    assert dim == 2 or dim == 3

    top = torch.cat((A, B.unsqueeze(1)), dim=1)
    if dim == 2:
        bottom = torch.tensor([[0.0, 0.0, 1.0]], dtype=A.dtype)
    else:
        bottom = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=A.dtype)

    return torch.cat((top, bottom), dim=0)


class TransformBase:
    "Abstract base class for 2D transforms"

    def direct_points(self, points: Tensor) -> Tensor:
        raise NotImplementedError

    def direct_vectors(self, vectors: Tensor) -> Tensor:
        raise NotImplementedError

    def inverse_points(self, points: Tensor) -> Tensor:
        raise NotImplementedError

    def inverse_vectors(self, vectors: Tensor) -> Tensor:
        raise NotImplementedError

    def hom_matrix(self) -> Tensor:
        "Homogenous transform matrix"
        raise NotImplementedError


class TranslateTransform(TransformBase):
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

    def hom_matrix(self) -> Tensor:
        dim = self.T.shape[0]
        A = torch.eye(dim, dtype=self.T.dtype)
        B = self.T
        return hom_matrix(A, B)


class LinearTransform(TransformBase):
    "Linear 2D transform: Y = AX"

    def __init__(self, A: Tensor, A_inv: Tensor):
        assert A.shape == A_inv.shape
        assert A.shape[0] == A.shape[1]
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

    def hom_matrix(self) -> Tensor:
        dim = self.A.shape[0]
        A = self.A
        B = torch.zeros((dim,), dtype=self.A.dtype)
        return hom_matrix(A, B)


class SurfaceExtentTransform(TransformBase):
    "Translation from a surface extent point"

    def __init__(self, surface: LocalSurface, dim: int):
        self.surface = surface
        self.dim = dim

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

    def hom_matrix(self) -> Tensor:
        dim = self.dim
        A = torch.eye(dim, dtype=self.surface.dtype)
        B = -self._extent()
        return hom_matrix(A, B)


class ComposeTransform(TransformBase):
    "Compose a list of 2D transforms"

    def __init__(self, transforms: list[TransformBase]):
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

    def hom_matrix(self) -> Tensor:
        return functools.reduce(
            lambda t1, t2: t2 @ t1, [t.hom_matrix() for t in self.transforms]
        )


def rotation_matrix_2D(
    theta : Tensor
) -> Tensor:
    theta = torch.atleast_1d(theta)
    return torch.vstack(
        (
            torch.cat((torch.cos(theta), -torch.sin(theta))),
            torch.cat((torch.sin(theta), torch.cos(theta))),
        )
    )


def basic_transform(
    scale: float,
    anchor: str,
    theta: float,
    translate: Iterable[float],
    dtype=torch.float64,
):
    """
    Experimental

    Create a transform Y = RS(X - A) + T
    Returns a function foo(surface)
    """

    def makeit(surface):
        # anchor
        transforms = [SurfaceExtentTransform(surface, 2)] if anchor == "extent" else []

        # scale
        transforms.append(
            LinearTransform(
                torch.tensor([[scale, 0.0], [0.0, scale]], dtype=dtype),
                torch.tensor([[1 / scale, 0.0], [0.0, 1 / scale]], dtype=dtype),
            )
        )

        # rotate
        transforms.append(
            LinearTransform(
                rotation_matrix_2D(torch.as_tensor(theta, dtype=dtype)),
                rotation_matrix_2D(-torch.as_tensor(theta, dtype=dtype)),
            )
        )

        # translate
        transforms.append(TranslateTransform(torch.as_tensor(translate, dtype=dtype)))

        return ComposeTransform(transforms)

    return makeit
