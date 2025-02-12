import numpy as np
import torch
from torchlensmaker.core.interp1d import interp1d


def test_grad() -> None:
    X = torch.tensor([0.0, 10.0, 20.0])
    Y = torch.tensor([0.0, -2.0, -3.0], requires_grad=True)

    newX = torch.tensor([7.0, 12.0, 0.0, 20.0])

    newY = interp1d(X, Y, newX)

    loss = newY.sum()

    grads = torch.autograd.grad(loss, (Y,))

    assert all((torch.isfinite(g).all() for g in grads))


def test_equals_numpy() -> None:
    X = torch.tensor([0.0, 10.0, 20.0])
    Y = torch.tensor([0.0, -2.0, -3.0])
    newX = torch.tensor([7.0, 12.0, 0.0, 20.0])

    newY = interp1d(X, Y, newX)
    newY_numpy = np.interp(newX, X, Y)

    assert np.allclose(newY_numpy, newY.numpy())
