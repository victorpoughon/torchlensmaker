import torch
import functools


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
    "Abstract base class for transforms"

    def __init__(self, dim: int, dtype: torch.dtype):
        self.dim = dim
        self.dtype = dtype

    def __repr__(self) -> str:
        return (
            f"[{type(self).__name__} {hex(id(self))} dim={self.dim} dtype={self.dtype}]"
        )

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


class IdentityTransform(TransformBase):
    def direct_points(self, points: Tensor) -> Tensor:
        return points

    def direct_vectors(self, vectors: Tensor) -> Tensor:
        return vectors

    def inverse_points(self, points: Tensor) -> Tensor:
        return points

    def inverse_vectors(self, vectors: Tensor) -> Tensor:
        return vectors

    def hom_matrix(self) -> Tensor:
        return hom_matrix(
            torch.eye(self.dim, dtype=self.dtype),
            torch.zeros((self.dim,), dtype=self.dtype),
        )


class TranslateTransform(TransformBase):
    "Translation transform Y = X + T"

    def __init__(self, T: Tensor):
        super().__init__(T.shape[0], T.dtype)
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
        A = torch.eye(self.dim, dtype=self.dtype)
        B = self.T
        return hom_matrix(A, B)


class LinearTransform(TransformBase):
    "Linear transform Y = AX"

    def __init__(self, A: Tensor, A_inv: Tensor):
        assert A.shape == A_inv.shape
        assert A.shape[0] == A.shape[1]
        super().__init__(A.shape[0], A.dtype)
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
        A = self.A
        B = torch.zeros((self.dim,), dtype=self.dtype)
        return hom_matrix(A, B)


class ComposeTransform(TransformBase):
    "Compose a list of transforms"

    def __init__(self, transforms: list[TransformBase]):
        assert len(transforms) > 0
        assert sum([t.dtype == transforms[0].dtype for t in transforms]) == len(
            transforms
        )
        assert sum([t.dim == transforms[0].dim for t in transforms]) == len(transforms)
        super().__init__(transforms[0].dim, transforms[0].dtype)
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


def forward_kinematic(transforms: list[TransformBase]) -> Tensor:
    "Compose transforms that describe a forward kinematic chain"

    return ComposeTransform(list(reversed(transforms)))
