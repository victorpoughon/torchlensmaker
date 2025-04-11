# This file is part of Torch Lens Maker
# Copyright (C) 2025 Victor Poughon
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

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
    def g(self, r: Tensor, tau: Tensor) -> Tensor:
        """
        2D sag function $g(r)$

        Args:
        * r: batched tensor of shape (...)
        * tau: normalization factor (0-dim)

        Returns:
        * batched tensor of shape (...)
        """
        raise NotImplementedError

    def g_grad(self, r: Tensor, tau: Tensor) -> Tensor:
        """
        Derivative of the 2D sag function $g'(r)$

        Args:
        * r: batched tensor of shape (...)
        * tau: normalization factor (0-dim)

        Returns:
        * batched tensor of shape (...)
        """
        raise NotImplementedError

    def G(self, y: Tensor, z: Tensor, tau: Tensor) -> Tensor:
        """
        3D sag function $G(X, Y) = g(\\sqrt{y^2 + z^2})$

        Args:
        * y: batched tensor of shape (...)
        * z: batched tensor of shape (...)
        * tau: normalization factor (0-dim)

        Returns:
        * batched tensor of shape (...)
        """
        raise NotImplementedError

    def G_grad(self, y: Tensor, z: Tensor, tau: Tensor) -> tuple[Tensor, Tensor]:
        """
        Gradient of the 3D sag function $\\nabla G(y, z)$

        Args:
        * y: batched tensor of shape (...)
        * z: batched tensor of shape (...)
        * tau: normalization factor (0-dim)

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

    def g(self, r: Tensor, tau: Tensor) -> Tensor:
        C = self.C
        r2 = torch.pow(r, 2)
        return safe_div(C * r2, 1 + safe_sqrt(1 - r2 * torch.pow(C, 2)))

    def g_grad(self, r: Tensor, tau: Tensor) -> Tensor:
        C = self.C
        return safe_div(C * r, safe_sqrt(1 - torch.pow(r, 2) * torch.pow(C, 2)))

    def G(self, y: Tensor, z: Tensor, tau: Tensor) -> Tensor:
        C = self.C
        r2 = torch.pow(y, 2) + torch.pow(z, 2)
        return safe_div(C * r2, 1 + safe_sqrt(1 - r2 * torch.pow(C, 2)))

    def G_grad(self, y: Tensor, z: Tensor, tau: Tensor) -> tuple[Tensor, Tensor]:
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

    def g(self, r: Tensor, tau: Tensor) -> Tensor:
        return torch.mul(self.A, torch.pow(r, 2))

    def g_grad(self, r: Tensor, tau: Tensor) -> Tensor:
        return 2 * self.A * r

    def G(self, y: Tensor, z: Tensor, tau: Tensor) -> Tensor:
        return torch.mul(self.A, (y**2 + z**2))

    def G_grad(self, y: Tensor, z: Tensor, tau: Tensor) -> tuple[Tensor, Tensor]:
        return 2 * self.A * y, 2 * self.A * z

    def to_dict(self, _dim: int) -> dict[str, Any]:
        return {"sag-type": "parabolic", "A": self.A.item()}


class Conical(SagFunction):
    def __init__(
        self,
        C: torch.Tensor,
        K: torch.Tensor,
        normalize: bool = False,
    ):
        assert C.dim() == 0
        assert K.dim() == 0
        self.C = C
        self.K = K
        self.normalize = normalize

    def parameters(self) -> dict[str, nn.Parameter]:
        return {
            name: param
            for name, param in {"C": self.C, "K": self.K}.items()
            if isinstance(param, nn.Parameter)
        }

    def unnorm(self, tau: Tensor) -> tuple[Tensor, Tensor]:
        assert tau.dim() == 0
        if self.normalize:
            return self.C / tau, self.K
        else:
            return self.C, self.K

    def g(self, r: Tensor, tau: Tensor) -> Tensor:
        r2 = torch.pow(r, 2)
        C, K = self.unnorm(tau)
        C2 = torch.pow(C, 2)
        return torch.div(C * r2, 1 + torch.sqrt(1 - (1 + K) * r2 * C2))

    def g_grad(self, r: Tensor, tau: Tensor) -> Tensor:
        r2 = torch.pow(r, 2)
        C, K = self.unnorm(tau)
        C2 = torch.pow(C, 2)

        return torch.div(C * r, torch.sqrt(1 - (1 + K) * r2 * C2))

    def G(self, y: Tensor, z: Tensor, tau: Tensor) -> Tensor:
        C, K = self.unnorm(tau)
        C2 = torch.pow(C, 2)
        r2 = y**2 + z**2

        return torch.div(C * r2, 1 + torch.sqrt(1 - (1 + K) * r2 * C2))

    def G_grad(self, y: Tensor, z: Tensor, tau: Tensor) -> tuple[Tensor, Tensor]:
        C, K = self.unnorm(tau)
        C2 = torch.pow(C, 2)
        r2 = y**2 + z**2

        denom = torch.sqrt(1 - (1 + K) * r2 * C2)

        return (C * y) / denom, (C * z) / denom

    def to_dict(self, _dim: int) -> dict[str, Any]:
        return {
            "sag-type": "conical",
            "K": self.K.item(),
            "C": self.C.item(),
            **({"normalize": self.normalize} if self.normalize else {}),
        }


def vbroad(vector: Tensor, base: int) -> Tensor:
    """
    Broadcasts a 1D tensor to be compatible with the dimensions of a batched tensor.

    Args:
    * base: Number of dimensions of some batched tensor of shape (...)
    * vector: A 1D tensor of shape (M)

    Returns:
    * A view of the vector tensor with shape (M, ...) that is broadcastable with the base tensor.
    """
    assert vector.dim() == 1
    return vector.view(vector.shape[0], *([1] * base))


class Aspheric(SagFunction):
    """
    Aspheric coefficient polynomial of the form:
    $c_0 r^4 + c_1 r^6 + c_2 r^8 + ...$
    """

    def __init__(self, coefficients: torch.Tensor, normalize: bool = False):
        assert coefficients.dim() == 1
        self.coefficients = coefficients
        self.normalize = normalize
        self.i = torch.arange(len(coefficients))  # indexing of coefficients

    def unnorm(self, tau: Tensor) -> Tensor:
        """
        Computes the unnormalized coefficients of the sag function given the
        normalization factor tau if normalization is enabled.

        When normalizion is enabled, the internal coefficients of the sag
        function are stored in their normalized form.
        """

        assert tau.dim() == 0
        if self.normalize:
            i = self.i
            return self.coefficients / torch.pow(tau, 3 + 2 * i)
        else:
            return self.coefficients

    def parameters(self) -> dict[str, nn.Parameter]:
        return (
            {"coefficients": self.coefficients}
            if isinstance(self.coefficients, nn.Parameter)
            else {}
        )

    def g(self, r: Tensor, tau: Tensor) -> Tensor:
        alphas = vbroad(self.unnorm(tau), r.dim())
        i = vbroad(self.i, r.dim())

        return torch.sum(alphas * torch.pow(r, 4 + 2 * i), dim=0)

    def g_grad(self, r: Tensor, tau: Tensor) -> Tensor:
        alphas = vbroad(self.unnorm(tau), r.dim())
        i = vbroad(self.i, r.dim())

        return torch.sum(alphas * (4 + 2 * i) * torch.pow(r, 3 + 2 * i), dim=0)

    def G(self, y: Tensor, z: Tensor, tau: Tensor) -> Tensor:
        r2 = y**2 + z**2
        alphas = vbroad(self.unnorm(tau), r2.dim())
        i = vbroad(self.i, r2.dim())

        return torch.sum(alphas * torch.pow(r2, 2 + i), dim=0)

    def G_grad(self, y: Tensor, z: Tensor, tau: Tensor) -> tuple[Tensor, Tensor]:
        r2 = y**2 + z**2
        alphas = vbroad(self.unnorm(tau), r2.dim())
        i = vbroad(self.i, r2.dim())

        coeffs_term = torch.sum(alphas * (4 + 2 * i) * torch.pow(r2, 1 + i), dim=0)

        return y * coeffs_term, z * coeffs_term

    def to_dict(self, _dim: int) -> dict[str, Any]:
        return {
            "sag-type": "aspheric",
            "coefficients": self.coefficients.tolist(),
            **({"normalize": self.normalize} if self.normalize else {}),
        }


class SagSum(SagFunction):
    """
    Sag function that is the sum of other sag functions
    """

    def __init__(self, terms):
        assert all((isinstance(a, SagFunction) for a in terms))
        self.terms = terms

    def parameters(self) -> dict[str, nn.Parameter]:
        return {
            f"{str(i)}_{name}": p
            for i, t in enumerate(self.terms)
            for name, p in t.parameters().items()
        }

    def to_dict(self, dim: int) -> dict[str, Any]:
        return {"sag-type": "sum", "terms": [t.to_dict(dim) for t in self.terms]}

    def g(self, r: Tensor, tau: Tensor) -> Tensor:
        return torch.sum(torch.stack([t.g(r, tau) for t in self.terms], dim=0), dim=0)

    def g_grad(self, r: Tensor, tau: Tensor) -> Tensor:
        return torch.sum(
            torch.stack([t.g_grad(r, tau) for t in self.terms], dim=0), dim=0
        )

    def G(self, y: Tensor, z: Tensor, tau: Tensor) -> Tensor:
        return torch.sum(
            torch.stack([t.G(y, z, tau) for t in self.terms], dim=0), dim=0
        )

    def G_grad(self, y: Tensor, z: Tensor, tau: Tensor) -> tuple[Tensor, Tensor]:
        grads = [t.G_grad(y, z, tau) for t in self.terms]
        grad_y = torch.sum(torch.stack([g[0] for g in grads], dim=0), dim=0)
        grad_z = torch.sum(torch.stack([g[1] for g in grads], dim=0), dim=0)
        return grad_y, grad_z
