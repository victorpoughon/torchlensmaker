# This file is part of Torch Lens Maker
# Copyright (C) 2024-present Victor Poughon
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

import math
from dataclasses import dataclass
from typing import Any, Callable, Optional, TypeAlias

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim

from torchlensmaker.kinematics.homogeneous_geometry import hom_identity
from torchlensmaker.sequential.sequential_data import SequentialData
from torchlensmaker.types import ScalarTensor

Tensor = torch.Tensor
RegularizationFunction = Callable[[nn.Module], Tensor]


def get_all_gradients(model: nn.Module) -> Tensor:
    grads = []
    for param in model.parameters():
        if param.grad is not None:
            grads.append(param.grad.view(-1))
    return torch.cat(grads)


# Alias for convinience
optim: TypeAlias = torch.optim


@dataclass
class OptimizationRecord:
    num_iter: int
    parameters: dict[str, list[torch.Tensor]]
    loss: torch.Tensor
    optics: nn.Module

    def plot(self) -> None:
        plot_optimization_record(self)

    def best(self) -> None:
        best_loss, idx = self.loss.min(dim=0)
        print(f"Best loss {best_loss.item()} at iteration {idx + 1} / {self.num_iter}")
        for n, p in self.parameters.items():
            print("   ", n, p[idx])
        print()


class ImagingModel(nn.Module):
    def __init__(self, optics, target):
        super().__init__()
        self.optics = optics
        self.target = target

    def forward(self, data: SequentialData) -> ScalarTensor:
        outputs = self.optics(data)
        out = self.target(outputs.rays, outputs.fk)
        return out.loss


def optimize(
    model: nn.Module,
    input_rays: SequentialData,
    target: nn.Module,
    optimizer: optim.Optimizer,
    num_iter: int,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
    regularization: Optional[RegularizationFunction] = None,
    nshow: int = 20,
) -> OptimizationRecord:
    if dtype is None:
        dtype = torch.get_default_dtype()

    if device is None:
        device = torch.get_default_device()

    # Record values for analysis
    parameters_record: dict[str, list[Tensor]] = {
        n: [] for n, _ in model.named_parameters()
    }
    loss_record = torch.zeros(num_iter)

    # We assume the last element is the loss element
    imaging_model = ImagingModel(model, target)

    show_every = math.ceil(num_iter / nshow)

    for i in range(num_iter):
        optimizer.zero_grad()

        # Evaluate the model
        loss = imaging_model(input_rays)

        # Add regularization function term
        if regularization is not None:
            loss = loss + regularization(model)

        if not loss.requires_grad:
            raise RuntimeError(
                "No differentiable loss computed by optical stack (loss.requires_grad is False)"
            )

        loss.backward()

        # Record loss values
        loss_record[i] = loss.detach()

        # Record parameter values
        for n, param in model.named_parameters():
            parameters_record[n].append(param.detach().clone())

        # Compute gradients for gradient magnitude
        # and sanity check that gradient isn't nan or inf
        grad = get_all_gradients(model)
        if torch.isnan(grad).any():
            print("ERROR: nan in grad", grad)
            raise RuntimeError(
                "nan in gradient, check your torch.where() (https://docs.jax.dev/en/latest/faq.html#gradients-contain-nan-where-using-where))"
            )

        optimizer.step()

        if i % show_every == 0 or i == num_iter - 1:
            iter_str = f"[{i + 1:>3}/{num_iter}]"
            L_str = (
                f"L= {loss.item():>6.5f} | grad norm= {torch.linalg.norm(grad):5>.4f}"
            )
            print(f"{iter_str} {L_str}")

    return OptimizationRecord(num_iter, parameters_record, loss_record, model)


def plot_optimization_record(record: OptimizationRecord) -> None:
    if record.num_iter == 0:
        return

    optics = record.optics
    parameters = record.parameters
    loss = record.loss

    # Plot parameters and loss
    fig, (ax1, ax2) = plt.subplots(2, 1)
    epoch_range = torch.arange(0, record.num_iter)
    ax2.plot(epoch_range, loss.detach(), label="loss")
    for n, param in optics.named_parameters():
        if parameters[n][0].dim() == 0:
            data = torch.stack(parameters[n]).detach().numpy()
            ax1.plot(epoch_range.detach(), data, label=n)
    ax1.set_title("parameters")
    ax1.legend()
    ax2.set_title("loss")
    ax2.legend()
    plt.show()
