import torch


def total_domain(points: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    return torch.full(points.shape[:-1], True, dtype=torch.bool, device=points.device)
