import torch


def implicit_xcylinder_2d(
    points: torch.Tensor, R: float | torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...


def implicit_xcylinder_3d(
    points: torch.Tensor, R: float | torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Implicit function for an infinite cylinder around the X axis
    """
    ...
