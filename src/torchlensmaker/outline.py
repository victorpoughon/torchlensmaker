import torch
import math


class Outline:
    "An outline limits the extent of a 3D surface in the local YZ plane"

    def contains(self, points: torch.Tensor, tol: float = 1e-6) -> torch.Tensor:
        raise NotImplementedError

    def max_radius(self) -> float:
        "Furthest distance to the X axis that's within the outline"
        raise NotImplementedError


class SquareOutline(Outline):
    "Square outline around the X axis"

    def __init__(self, side_length: float):
        self.side_length = side_length

    def contains(self, points: torch.Tensor, tol: float = 1e-6) -> torch.Tensor:
        return (
            torch.maximum(torch.abs(points[:, 1]), torch.abs(points[:, 2]))
            < self.side_length / 2
        )

    def max_radius(self) -> float:
        return math.sqrt(2) * self.side_length / 2


class CircularOutline(Outline):
    "Fixed distance to the X axis"

    def __init__(self, diameter: float):
        self.diameter = diameter

    def contains(self, points: torch.Tensor, tol: float = 1e-6) -> torch.Tensor:
        return torch.le(torch.hypot(points[:, 1], points[:, 2]), self.diameter / 2)

    def max_radius(self) -> float:
        return self.diameter / 2
