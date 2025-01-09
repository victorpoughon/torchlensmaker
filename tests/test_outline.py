import torch
import torchlensmaker as tlm
import pytest

import math


def test_square_outline() -> None:
    outline = tlm.SquareOutline(5.0)

    assert outline.max_radius() == pytest.approx(math.sqrt(2) * 5 / 2)

    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 2.4, 2.4],
            [0.0, 2.6, 2.0],
        ]
    )

    assert torch.all(outline.contains(points) == torch.tensor([True, True, True, True, False]))



def test_circular_outline() -> None:
    outline = tlm.CircularOutline(5.0)

    assert outline.max_radius() == 5.0 / 2

    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 2.4, 2.4],
            [0.0, 2.6, 2.0],
        ]
    )
    print(outline.contains(points))
    assert torch.all(outline.contains(points) == torch.tensor([True, True, True, False, False]))
