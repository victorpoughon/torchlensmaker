import torch
from torchlensmaker.surfaces import LocalSurface
import functools

from typing import Iterable

# for shorter type annotations
Tensor = torch.Tensor


def hom_matrix(A: Tensor, B: Tensor) -> Tensor:
    "Homogeneous 3x3 transform matrix for 2D transform AX+B"
    assert A.dim() == 2
    assert B.dim() == 1
    assert A.shape[0] == A.shape[1] == B.shape[0]
    assert A.dtype == B.dtype
    dim = A.shape[0]
    assert dim == 2 or dim == 3

    top = torch.cat((A, B.unsqueeze(0).T), dim=1)
    if dim == 2:
        bottom = torch.tensor([[0.0, 0.0, 1.0]], dtype=A.dtype)
    else:
        bottom = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=A.dtype)

    return torch.cat((top, bottom), dim=0)


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

    def hom_matrix(self) -> Tensor:
        "Homogenous transform matrix"
        raise NotImplementedError


class TranslateTransform2D(Transform2DBase):
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


class LinearTransform2D(Transform2DBase):
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


class SurfaceExtentTransform2D(Transform2DBase):
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


class ComposeTransform2D(Transform2DBase):
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


def rotation_matrix_2D(
    theta: float | Tensor, dtype: torch.dtype | None = None
) -> Tensor:
    if dtype is None:
        dtype = theta.dtype if isinstance(theta, Tensor) else torch.float64
    elif isinstance(theta, Tensor):
        assert dtype == theta.dtype

    theta = torch.atleast_1d(torch.as_tensor(theta, dtype=dtype))
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
        transforms = [SurfaceExtentTransform2D(surface)] if anchor == "extent" else []

        # scale
        transforms.append(
            LinearTransform2D(
                torch.tensor([[scale, 0.0], [0.0, scale]], dtype=dtype),
                torch.tensor([[1 / scale, 0.0], [0.0, 1 / scale]], dtype=dtype),
            )
        )

        # rotate
        transforms.append(
            LinearTransform2D(
                rotation_matrix_2D(theta, dtype=dtype),
                rotation_matrix_2D(-theta, dtype=dtype),
            )
        )

        # translate
        transforms.append(TranslateTransform2D(torch.as_tensor(translate, dtype=dtype)))

        return ComposeTransform2D(transforms)

    return makeit
