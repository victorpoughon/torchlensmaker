import math
import torch
from torchlensmaker.core.rot2d import rot2d


def test_single_vector_single_angle() -> None:
    v = torch.tensor([1.0, 0.0])
    theta = torch.tensor(torch.pi / 4)
    result = rot2d(v, theta)
    assert torch.allclose(
        result, torch.tensor([1.0 / math.sqrt(2), 1.0 / math.sqrt(2)])
    )


def test_single_vector_batch_angle() -> None:
    v = torch.tensor([1.0, 0.0])
    theta = torch.tensor([torch.pi / 4, torch.pi / 3, torch.pi / 2])
    result = rot2d(v, theta)

    print(result)

    assert torch.allclose(
        result,
        torch.tensor(
            [
                [1.0 / math.sqrt(2), 1.0 / math.sqrt(2)],
                [1 / 2, math.sqrt(3) / 2],
                [0.0, 1.0],
            ]
        ),
        rtol=1e-05,
        atol=1e-07,
    )


def test_batch_vectors_single_angle() -> None:
    v = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    theta = torch.tensor(torch.pi / 3)
    result = rot2d(v, theta)

    assert torch.allclose(
        result,
        torch.tensor(
            [
                [1 / 2, math.sqrt(3) / 2],
                [-math.sqrt(3) / 2, 1 / 2],
                [-1 / 2, -math.sqrt(3) / 2],
            ]
        ),
    )


def test_batch_vectors_batch_angles() -> None:
    v = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    theta = torch.tensor([-torch.pi / 3, torch.pi / 4, torch.pi / 2])
    result = rot2d(v, theta)

    expected = torch.tensor(
        [
            [1 / 2, -math.sqrt(3) / 2],
            [-1.0 / math.sqrt(2), 1.0 / math.sqrt(2)],
            [0.0, -1.0],
        ]
    )

    assert torch.allclose(result, expected, rtol=1e-05, atol=1e-07)
