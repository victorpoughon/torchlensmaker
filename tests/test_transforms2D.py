import pytest
import typing

import torch

from torchlensmaker.transforms2D import (
    Transform2DBase,
    Translate2D,
    Linear2D,
    ComposeTransform2D,
    SurfaceExtent2D,
    rotation_matrix_2D,
)

from torchlensmaker.surfaces import (
    Sphere,
    Parabola,
    SquarePlane,
    CircularPlane,
)


@pytest.fixture(params=[torch.float32, torch.float64], ids=["float32", "float64"])
def dtype_fixture(request: pytest.FixtureRequest) -> typing.Any:
    return request.param


@pytest.fixture
def make_transforms(
    dtype_fixture: torch.dtype,
) -> tuple[torch.dtype, list[Transform2DBase]]:
    dtype = dtype_fixture
    # Translate2D
    T_id = Translate2D(torch.tensor([0.0, 0.0], dtype=dtype))
    T_1 = Translate2D(torch.tensor([1.0, 1.0], dtype=dtype))
    T_rand = Translate2D(torch.rand((2,), dtype=dtype))

    # Linear2D
    L_id = Linear2D(torch.eye(2, dtype=dtype), torch.eye(2, dtype=dtype))
    L_scale = Linear2D(
        torch.diag(torch.tensor([0.5, 0.25], dtype=dtype)),
        torch.diag(torch.tensor([2.0, 4.0], dtype=dtype)),
    )
    theta = torch.as_tensor(1.5, dtype=dtype)
    L_rot = Linear2D(
        torch.tensor(
            [
                [torch.cos(theta), -torch.sin(theta)],
                [torch.sin(theta), torch.cos(theta)],
            ]
        ),
        torch.tensor(
            [
                [torch.cos(-theta), -torch.sin(-theta)],
                [torch.sin(-theta), torch.cos(-theta)],
            ]
        ),
    )

    # SurfaceExtent2D
    s1 = Sphere(35.0, 35 / 2, dtype=dtype)
    s2 = Parabola(35.0, 0.010, dtype=dtype)
    s3 = SquarePlane(35.0, dtype=dtype)
    s4 = CircularPlane(35.0, dtype=dtype)
    A_1 = SurfaceExtent2D(s1)
    A_2 = SurfaceExtent2D(s2)
    A_3 = SurfaceExtent2D(s3)
    A_4 = SurfaceExtent2D(s4)

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

    return (
        dtype,
        [
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
        ],
    )


def check_roundtrip(t: Transform2DBase, dtype: torch.dtype) -> None:
    N = 5

    if dtype == torch.float32:
        atol, rtol = 1e-5, 1e-4
    elif dtype == torch.float64:
        atol, rtol = 1e-10, 1e-8

    points = torch.rand((N, 2), dtype=dtype)
    rtp1 = t.direct_points(t.inverse_points(points))
    rtp2 = t.inverse_points(t.direct_points(points))
    assert torch.allclose(rtp1, points, atol=atol, rtol=rtol), rtp1 - points
    assert torch.allclose(rtp2, points, atol=atol, rtol=rtol), rtp2 - points

    vectors = torch.rand((N, 2), dtype=dtype)
    rtv1 = t.direct_vectors(t.inverse_vectors(vectors))
    rtv2 = t.inverse_vectors(t.direct_vectors(vectors))
    assert torch.allclose(rtv1, vectors, atol=atol, rtol=rtol), rtv1 - vectors
    assert torch.allclose(rtv2, vectors, atol=atol, rtol=rtol), rtv2 - vectors


def check_preserve_dtype(t: Transform2DBase, dtype: torch.dtype) -> None:
    N = 1

    assert t.direct_points(torch.rand((N, 2), dtype=dtype)).dtype == dtype
    assert t.direct_vectors(torch.rand((N, 2), dtype=dtype)).dtype == dtype
    assert t.inverse_points(torch.rand((N, 2), dtype=dtype)).dtype == dtype
    assert t.inverse_vectors(torch.rand((N, 2), dtype=dtype)).dtype == dtype
    assert t.matrix3().dtype == dtype


def test_roundtrip(make_transforms: tuple[torch.dtype, list[Transform2DBase]]) -> None:
    dtype, transforms = make_transforms
    for t in transforms:
        check_roundtrip(t, dtype)


def test_preserve_dtype(
    make_transforms: tuple[torch.dtype, list[Transform2DBase]]
) -> None:
    dtype, transforms = make_transforms
    for t in transforms:
        check_preserve_dtype(t, dtype)


def check_matrix3(t: Transform2DBase, dtype: torch.dtype) -> None:
    N = 5

    if dtype == torch.float32:
        atol, rtol = 1e-5, 1e-4
    elif dtype == torch.float64:
        atol, rtol = 1e-10, 1e-8

    # cartesian and homogeneous coordinates
    points = torch.rand((N, 2), dtype=dtype)
    hom_points = torch.column_stack(
        (points, torch.ones((points.shape[0], 1), dtype=dtype))
    )

    # direct transform using homogenous coordinates
    f_hom = (t.matrix3() @ hom_points.T).T

    # direct transform using cartesian coordinates
    f_direct = t.direct_points(points)

    # convert to homogenous after application of the direct transform
    f_direct_hom = torch.column_stack(
        (f_direct, torch.ones((points.shape[0], 1), dtype=dtype))
    )

    assert torch.allclose(f_direct_hom, f_hom, atol=atol, rtol=rtol), (
        f_hom - f_direct_hom
    )


def test_matrix3(make_transforms: tuple[torch.dtype, list[Transform2DBase]]) -> None:
    dtype, transforms = make_transforms
    for t in transforms:
        check_matrix3(t, dtype)


def test_grad_translate2D(dtype_fixture: torch.dtype) -> None:
    dtype = dtype_fixture

    transform = Translate2D(torch.tensor([0.0, 0.0], dtype=dtype, requires_grad=True))

    loss = transform.matrix3().sum()
    loss.backward()  # type: ignore[no-untyped-call]
    grad = transform.T.grad

    assert grad is not None

    # known grad for a translation
    assert torch.allclose(grad, torch.ones(2, dtype=dtype))

    assert torch.all(torch.isfinite(grad))
    assert grad.dtype == dtype


def test_grad_scale2D(dtype_fixture: torch.dtype) -> None:
    dtype = dtype_fixture

    scale = torch.tensor([5.0], dtype=dtype, requires_grad=True)
    transform = Linear2D(
        torch.eye(2, dtype=dtype) * scale, torch.eye(2, dtype=dtype) * 1.0 / scale
    )

    loss = transform.matrix3().sum()
    loss.backward()  # type: ignore[no-untyped-call]
    grad = scale.grad

    assert grad is not None

    # known grad for a scale
    assert torch.allclose(grad, torch.tensor([2.0], dtype=dtype))

    assert torch.all(torch.isfinite(grad))
    assert grad.dtype == dtype


def test_grad_rot2D(dtype_fixture: torch.dtype) -> None:
    dtype = dtype_fixture

    theta = torch.tensor([0.1], dtype=dtype, requires_grad=True)
    transform = Linear2D(rotation_matrix_2D(theta), rotation_matrix_2D(-theta))

    loss = transform.matrix3().sum()
    loss.backward()  # type: ignore[no-untyped-call]
    grad = theta.grad

    assert grad is not None
    assert torch.all(torch.isfinite(grad))
    assert grad.dtype == dtype


# TODO implement test that invalid initializations raises error, like Linear2D(torch.zeros(), ...)
