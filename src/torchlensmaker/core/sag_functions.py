import torch
import torch.nn as nn

from typing import TypeAlias, Any

Tensor: TypeAlias = torch.Tensor


def safe_sqrt(radicand: Tensor) -> Tensor:
    """
    Gradient safe version of torch.sqrt() that returns 0 where radicand <= 0
    """
    ok = radicand > 0
    safe = torch.zeros_like(radicand)
    return torch.sqrt(torch.where(ok, radicand, safe))


def safe_div(dividend: Tensor, divisor: Tensor) -> Tensor:
    """
    Gradient safe version of torch.div() that returns dividend where divisor == 0
    """

    ok = divisor != torch.zeros((), dtype=divisor.dtype)
    safe = torch.ones_like(divisor)
    return torch.div(dividend, torch.where(ok, divisor, safe))


class SagFunction:
    def g(self, r: Tensor) -> Tensor:
        """
        2D sag function $g(r)$

        Args:
        * r: batched tensor of shape (...)

        Returns:
        * batched tensor of shape (...)
        """
        raise NotImplementedError

    def g_grad(self, r: Tensor) -> Tensor:
        """
        Derivative of the 2D sag function $g'(r)$

        Args:
        * r: batched tensor of shape (...)

        Returns:
        * batched tensor of shape (...)
        """
        raise NotImplementedError

    def G(self, y: Tensor, z: Tensor) -> Tensor:
        """
        3D sag function $G(X, Y) = g(\\sqrt{y^2 + z^2})$

        Args:
        * y: batched tensor of shape (...)
        * z: batched tensor of shape (...)

        Returns:
        * batched tensor of shape (...)
        """
        raise NotImplementedError

    def G_grad(self, y: Tensor, z: Tensor) -> tuple[Tensor, Tensor]:
        """
        Gradient of the 3D sag function $\\nabla G(y, z)$

        Args:
        * y: batched tensor of shape (...)
        * z: batched tensor of shape (...)

        Returns:
        * grad_y: batched tensor of shape (...)
        * grad_z: batched tensor of shape (...)
        """
        raise NotImplementedError

    def to_dict(self, dim: int) -> dict[str, Any]:
        raise NotImplementedError


class Spherical(SagFunction):
    def __init__(self, C: torch.Tensor | nn.Parameter):
        assert C.dim() == 0
        self.C = C

    def parameters(self) -> dict[str, nn.Parameter]:
        return {"C": self.C} if isinstance(self.C, nn.Parameter) else {}

    def g(self, r: Tensor) -> Tensor:
        C = self.C
        r2 = torch.pow(r, 2)
        return safe_div(C * r2, 1 + safe_sqrt(1 - r2 * torch.pow(C, 2)))

    def g_grad(self, r: Tensor) -> Tensor:
        C = self.C
        return safe_div(C * r, safe_sqrt(1 - torch.pow(r, 2) * torch.pow(C, 2)))

    def G(self, y: Tensor, z: Tensor) -> Tensor:
        C = self.C
        r2 = torch.pow(y, 2) + torch.pow(z, 2)
        return safe_div(C * r2, 1 + safe_sqrt(1 - r2 * torch.pow(C, 2)))

    def G_grad(self, y: Tensor, z: Tensor) -> tuple[Tensor, Tensor]:
        C = self.C
        r2 = torch.pow(y, 2) + torch.pow(z, 2)
        denom = safe_sqrt(1 - r2 * torch.pow(C, 2))
        return safe_div(y * C, denom), safe_div(z * C, denom)

    def to_dict(self, _dim: int) -> dict[str, Any]:
        return {"sag-type": "spherical", "C": self.C.item()}


class Parabolic(SagFunction):
    def __init__(self, A: torch.Tensor | nn.Parameter):
        assert A.dim() == 0
        self.A = A

    def parameters(self) -> dict[str, nn.Parameter]:
        return {"A": self.A} if isinstance(self.A, nn.Parameter) else {}

    def g(self, r: Tensor) -> Tensor:
        return torch.mul(self.A, torch.pow(r, 2))

    def g_grad(self, r: Tensor) -> Tensor:
        return 2 * self.A * r

    def G(self, y: Tensor, z: Tensor) -> Tensor:
        return torch.mul(self.A, (y**2 + z**2))

    def G_grad(self, y: Tensor, z: Tensor) -> tuple[Tensor, Tensor]:
        return 2 * self.A * y, 2 * self.A * z

    def to_dict(self, _dim: int) -> dict[str, Any]:
        return {"sag-type": "parabolic", "A": self.A.item()}


# TODO split into conical and aspheric coefficients
class Aspheric(SagFunction):
    def __init__(
        self,
        C: torch.Tensor | nn.Parameter,
        K: torch.Tensor | nn.Parameter,
        A4: torch.Tensor | nn.Parameter,
    ):
        assert C.dim() == 0
        assert K.dim() == 0
        assert A4.dim() == 0

        self.C = C
        self.K = K
        self.A4 = A4

    def parameters(self) -> dict[str, nn.Parameter]:
        possible = {
            "C": self.C,
            "K": self.K,
            "A4": self.A4,
        }
        return {
            name: value
            for name, value in possible.items()
            if isinstance(value, nn.Parameter)
        }

    def g(self, r: Tensor) -> Tensor:
        r2 = torch.pow(r, 2)
        K, C, A4 = self.K, self.C, self.A4
        C2 = torch.pow(C, 2)
        return torch.div(
            C * r2, 1 + torch.sqrt(1 - (1 + K) * r2 * C2)
        ) + A4 * torch.pow(r2, 2)

    def g_grad(self, r: Tensor) -> Tensor:
        r2 = torch.pow(r, 2)
        K, C, A4 = self.K, self.C, self.A4
        C2 = torch.pow(C, 2)

        return torch.div(C * r, torch.sqrt(1 - (1 + K) * r2 * C2)) + 4 * A4 * torch.pow(
            r, 3
        )

    def G(self, y: Tensor, z: Tensor) -> Tensor:
        K, C, A4 = self.K, self.C, self.A4
        C2 = torch.pow(C, 2)
        r2 = y**2 + z**2

        return torch.div(
            C * r2, 1 + torch.sqrt(1 - (1 + K) * r2 * C2)
        ) + A4 * torch.pow(r2, 2)

    def G_grad(self, y: Tensor, z: Tensor) -> tuple[Tensor, Tensor]:
        K, C, A4 = self.K, self.C, self.A4
        C2 = torch.pow(C, 2)
        r2 = y**2 + z**2

        denom = torch.sqrt(1 - (1 + K) * r2 * C2)
        coeffs_term = 4 * A4 * r2

        return (C * y) / denom + y * coeffs_term, (C * z) / denom + z * coeffs_term

    def to_dict(self, _dim: int) -> dict[str, Any]:
        return {} # TODO
