import torch
import torch.nn as nn

def parameter(data: float | int | torch.Tensor, dtype: torch.dtype = torch.float64) -> nn.Parameter:
    return nn.Parameter(torch.as_tensor(data, dtype=dtype))
