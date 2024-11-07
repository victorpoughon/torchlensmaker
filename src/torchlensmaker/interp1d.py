import torch

def interp1d(X, Y, newX):
    assert X.ndim == Y.ndim == 1

    # find intervals
    indices = torch.searchsorted(X, newX)

    # special case for newX == X[0]
    indices = torch.where(newX == X[0], 1, indices)

    # -1 here because we want the start of the interval
    indices = indices - 1
    
    # make sure all newX are within the X domain
    assert torch.min(indices) >= 0
    assert torch.max(indices) <= X.numel() - 1
    
    # compute slopes
    # careful potential div by zero here
    slopes = torch.diff(Y) / torch.diff(X)

    return Y[indices] + slopes[indices]*(newX - X[indices])
