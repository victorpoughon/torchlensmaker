import torch
import torch.nn as nn

def parameter(data: float | int | torch.Tensor) -> nn.Parameter:
    return nn.Parameter(torch.as_tensor(data, dtype=torch.float64))
