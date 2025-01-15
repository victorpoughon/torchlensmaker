import pytest

import torch

from torchlensmaker.transforms2D import (
    Transform2DBase,
    Translate2D,
    Linear2D,
    Compose2D,
    ComposeList2D,
    SurfaceExtent2D,
)

from torchlensmaker.surfaces import (
    Sphere,
    Parabola,
    SquarePlane,
    CircularPlane,
)


def make_transforms(dtype: torch.dtype) -> list[Transform2DBase]:
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

    # Compose2D
    C_1 = Compose2D(T_id, T_1)
    C_2 = Compose2D(T_id, L_id)
    C_3 = Compose2D(T_1, L_scale)
    C_4 = Compose2D(L_rot, L_scale)
    C_5 = Compose2D(C_4, T_1)
    C_6 = Compose2D(C_5, C_5)
    C_7 = Compose2D(C_4, C_2)
    C_8 = Compose2D(A_1, A_2)
    C_9 = Compose2D(A_1, C_6)
    C_10 = Compose2D(C_9, C_4)

    # ComposeList2D
    D_1 = ComposeList2D([T_id, T_1])
    D_2 = ComposeList2D([T_id, L_id])
    D_3 = ComposeList2D([T_1, L_scale])
    D_4 = ComposeList2D([L_rot, L_scale])
    D_5 = ComposeList2D([D_4, T_1])
    D_6 = ComposeList2D([D_5, D_5, D_2, A_3])
    D_7 = ComposeList2D([D_4, D_2])
    D_8 = ComposeList2D([A_1, A_2, C_2, C_5])
    D_9 = ComposeList2D([A_1, D_6])
    D_10 = ComposeList2D([D_9, D_4])

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
        C_1,
        C_2,
        C_3,
        C_4,
        C_5,
        C_6,
        C_7,
        C_8,
        C_9,
        C_10,
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


def check_roundtrip(t: Transform2DBase, dtype: torch.dtype) -> None:
    N = 50

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


@pytest.fixture
def transforms_float32() -> list[Transform2DBase]:
    return make_transforms(torch.float32)


@pytest.fixture
def transforms_float64() -> list[Transform2DBase]:
    return make_transforms(torch.float64)


def test_roundtrip_float32(transforms_float32: list[Transform2DBase]) -> None:
    for t in transforms_float32:
        check_roundtrip(t, torch.float32)


def test_roundtrip_float64(transforms_float64: list[Transform2DBase]) -> None:
    for t in transforms_float64:
        check_roundtrip(t, torch.float64)


def test_preserve_dtype_float32(transforms_float32: list[Transform2DBase]) -> None:
    for t in transforms_float32:
        check_preserve_dtype(t, torch.float32)


def test_preserve_dtype_float64(transforms_float64: list[Transform2DBase]) -> None:
    for t in transforms_float64:
        check_preserve_dtype(t, torch.float64)


def test_invalid() -> None:
    # TODO check that Linear(00) raise an error?
    ...


# TODO test matrix3 product compared with manual transform
# TODO check that we can parameterize and compute grads
