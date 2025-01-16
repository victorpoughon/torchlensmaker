import pytest
import typing

import torch

from torchlensmaker.transforms import (
    Transform2DBase,
    TranslateTransform2D,
    LinearTransform2D,
    ComposeTransform2D,
    SurfaceExtentTransform2D,
    rotation_matrix_2D,
)

from torchlensmaker.surfaces import (
    Sphere,
    Parabola,
    SquarePlane,
    CircularPlane,
)

from torchlensmaker.rot3d import euler_angles_to_matrix


@pytest.fixture(params=[torch.float32, torch.float64], ids=["float32", "float64"])
def dtype(request: pytest.FixtureRequest) -> typing.Any:
    return request.param


@pytest.fixture(params=[2, 3], ids=["2D", "3D"])
def dim(request: pytest.FixtureRequest) -> typing.Any:
    return request.param


@pytest.fixture
def make_transforms(
    dtype: torch.dtype,
    dim: int,
) -> tuple[torch.dtype, int, list[Transform2DBase]]:
    if dim == 2:
        return dtype, dim, make_transforms_2D(dtype)
    else:
        return dtype, dim, make_transforms_3D(dtype)


def make_transforms_2D(dtype: torch.dtype) -> list[Transform2DBase]:
    # Translate2D
    T_id = TranslateTransform2D(torch.tensor([0.0, 0.0], dtype=dtype))
    T_1 = TranslateTransform2D(torch.tensor([1.0, 1.0], dtype=dtype))
    T_rand = TranslateTransform2D(torch.rand((2,), dtype=dtype))

    # Linear2D
    L_id = LinearTransform2D(torch.eye(2, dtype=dtype), torch.eye(2, dtype=dtype))
    L_scale = LinearTransform2D(
        torch.diag(torch.tensor([0.5, 0.25], dtype=dtype)),
        torch.diag(torch.tensor([2.0, 4.0], dtype=dtype)),
    )
    theta = torch.as_tensor(1.5, dtype=dtype)
    L_rot = LinearTransform2D(rotation_matrix_2D(theta), rotation_matrix_2D(-theta))

    # SurfaceExtent2D
    s1 = Sphere(35.0, 35 / 2, dtype=dtype)
    s2 = Parabola(35.0, 0.010, dtype=dtype)
    s3 = SquarePlane(35.0, dtype=dtype)
    s4 = CircularPlane(35.0, dtype=dtype)
    A_1 = SurfaceExtentTransform2D(s1, 2)
    A_2 = SurfaceExtentTransform2D(s2, 2)
    A_3 = SurfaceExtentTransform2D(s3, 2)
    A_4 = SurfaceExtentTransform2D(s4, 2)

    # ComposeTransform2D
    D_1 = ComposeTransform2D([T_id, T_1])
    D_2 = ComposeTransform2D([T_id, L_id])
    D_3 = ComposeTransform2D([T_1, L_scale])
    D_4 = ComposeTransform2D([L_rot, L_scale])
    D_5 = ComposeTransform2D([D_4, T_1])
    D_6 = ComposeTransform2D([D_5, D_5, D_2, A_3])
    D_7 = ComposeTransform2D([D_4, D_2, D_2, D_5, D_5])
    D_8 = ComposeTransform2D([A_1, A_2, D_2, D_5])
    D_9 = ComposeTransform2D([A_1, D_6])
    D_10 = ComposeTransform2D([D_9, D_4])

    return [
        T_id,
        T_1,
        T_rand,
        L_id,
        L_scale,
        L_rot,
        A_1,
        A_2,
        A_3,
        A_4,
        D_1,
        D_2,
        D_3,
        D_4,
        D_5,
        D_6,
        D_7,
        D_8,
        D_9,
        D_10,
    ]


def make_transforms_3D(dtype: torch.dtype) -> list[Transform2DBase]:
    # Translate
    T_id = TranslateTransform2D(torch.tensor([0.0, 0.0, 0.0], dtype=dtype))

    # TODO more

    return [T_id]


def test_roundtrip(
    make_transforms: tuple[torch.dtype, int, list[Transform2DBase]]
) -> None:
    dtype, dim, transforms = make_transforms
    N = 5
    for t in transforms:

        if dtype == torch.float32:
            atol, rtol = 1e-5, 1e-4
        elif dtype == torch.float64:
            atol, rtol = 1e-10, 1e-8

        points = torch.rand((N, dim), dtype=dtype)
        rtp1 = t.direct_points(t.inverse_points(points))
        rtp2 = t.inverse_points(t.direct_points(points))
        assert torch.allclose(rtp1, points, atol=atol, rtol=rtol), rtp1 - points
        assert torch.allclose(rtp2, points, atol=atol, rtol=rtol), rtp2 - points

        vectors = torch.rand((N, dim), dtype=dtype)
        rtv1 = t.direct_vectors(t.inverse_vectors(vectors))
        rtv2 = t.inverse_vectors(t.direct_vectors(vectors))
        assert torch.allclose(rtv1, vectors, atol=atol, rtol=rtol), rtv1 - vectors
        assert torch.allclose(rtv2, vectors, atol=atol, rtol=rtol), rtv2 - vectors


def test_preserve_dtype(
    make_transforms: tuple[torch.dtype, int, list[Transform2DBase]]
) -> None:
    dtype, dim, transforms = make_transforms
    N = 3
    for t in transforms:
        assert t.direct_points(torch.rand((N, dim), dtype=dtype)).dtype == dtype
        assert t.direct_vectors(torch.rand((N, dim), dtype=dtype)).dtype == dtype
        assert t.inverse_points(torch.rand((N, dim), dtype=dtype)).dtype == dtype
        assert t.inverse_vectors(torch.rand((N, dim), dtype=dtype)).dtype == dtype
        assert t.hom_matrix().dtype == dtype


def test_hom_matrix(
    make_transforms: tuple[torch.dtype, int, list[Transform2DBase]]
) -> None:
    dtype, dim, transforms = make_transforms
    N = 5
    for t in transforms:
        if dtype == torch.float32:
            atol, rtol = 1e-5, 1e-4
        elif dtype == torch.float64:
            atol, rtol = 1e-10, 1e-8

        # cartesian and homogeneous coordinates
        points = torch.rand((N, dim), dtype=dtype)
        hom_points = torch.column_stack(
            (points, torch.ones((points.shape[0], 1), dtype=dtype))
        )

        # direct transform using homogenous coordinates
        f_hom = (t.hom_matrix() @ hom_points.T).T

        # direct transform using cartesian coordinates
        f_direct = t.direct_points(points)

        # convert to homogenous after application of the direct transform
        f_direct_hom = torch.column_stack(
            (f_direct, torch.ones((points.shape[0], 1), dtype=dtype))
        )

        assert t.hom_matrix().dtype == dtype
        assert f_hom.dtype == dtype
        assert f_direct.dtype == dtype
        assert f_direct_hom.dtype == dtype

        assert torch.allclose(f_direct_hom, f_hom, atol=atol, rtol=rtol), (
            f_hom - f_direct_hom
        )


def test_grad_translate2D(dtype: torch.dtype, dim: int) -> None:
    transform = TranslateTransform2D(
        torch.zeros((dim,), dtype=dtype, requires_grad=True)
    )

    loss = transform.hom_matrix().sum()
    loss.backward()  # type: ignore[no-untyped-call]
    grad = transform.T.grad

    assert grad is not None

    # known grad for a translation
    assert torch.allclose(grad, torch.ones(dim, dtype=dtype))

    assert torch.all(torch.isfinite(grad))
    assert grad.dtype == dtype


def test_grad_scale2D(dtype: torch.dtype, dim: int) -> None:
    scale = torch.tensor([5.0], dtype=dtype, requires_grad=True)
    transform = LinearTransform2D(
        torch.eye(dim, dtype=dtype) * scale, torch.eye(dim, dtype=dtype) * 1.0 / scale
    )

    loss = transform.hom_matrix().sum()
    loss.backward()  # type: ignore[no-untyped-call]
    grad = scale.grad

    assert grad is not None

    # known grad for a scale
    assert torch.allclose(grad, torch.tensor([dim], dtype=dtype))

    assert torch.all(torch.isfinite(grad))
    assert grad.dtype == dtype


def test_grad_rot2D(dtype: torch.dtype, dim: int) -> None:
    theta2 = torch.tensor([0.1], dtype=dtype, requires_grad=True)
    theta3 = torch.tensor([0.1, 0.2, 0.3], dtype=dtype, requires_grad=True)

    if dim == 2:
        M = rotation_matrix_2D(theta2)
    else:
        M = euler_angles_to_matrix(theta3, "XYZ")

    transform = LinearTransform2D(M, M.T)

    loss = transform.hom_matrix().sum()
    loss.backward()  # type: ignore[no-untyped-call]

    if dim == 2:
        grad = theta2.grad
    else:
        grad = theta3.grad

    assert grad is not None
    assert torch.all(torch.isfinite(grad))
    assert grad.dtype == dtype


# TODO implement test that invalid initializations raises error, like Linear2D(torch.zeros(), ...)
