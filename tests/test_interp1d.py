import numpy as np
import torch
from torchlensmaker.interp1d import interp1d


def test_grad():
    X = torch.tensor([0., 10., 20.])
    Y = torch.tensor([0., -2., -3.], requires_grad=True)

    newX = torch.tensor([7., 12., 0., 20.])

    newY = interp1d(X, Y, newX)

    loss = newY.sum()

    grads = torch.autograd.grad(loss, (Y,))
    
    assert all(
        (torch.isfinite(g).all() for g in grads)
    )


def test_equals_numpy():
    X = torch.tensor([0., 10., 20.])
    Y = torch.tensor([0., -2., -3.])
    newX = torch.tensor([7., 12., 0., 20.])

    newY = interp1d(X, Y, newX)
    newY_numpy = np.interp(newX, X, Y)

    assert np.allclose(newY_numpy, newY.numpy())
