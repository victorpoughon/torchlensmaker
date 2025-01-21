import torch
import torch.nn as nn
import torch.optim as optim

import math
from dataclasses import dataclass

import matplotlib.pyplot as plt

import torchlensmaker as tlm

from typing import Any

Tensor = torch.Tensor


def get_all_gradients(model: nn.Module) -> Tensor:
    grads = []
    for param in model.parameters():
        if param.grad is not None:
            grads.append(param.grad.view(-1))
    return torch.cat(grads)


@dataclass
class OptimizationRecord:
    num_iter: int
    parameters: dict[str, list[torch.Tensor]]
    loss: torch.Tensor
    optics: nn.Module

    def plot(self) -> None:
        plot_optimization_record(self)


def optimize(
    optics: nn.Module,
    optimizer: optim.Optimizer,
    sampling: dict[str, Any],
    num_iter: int,
    nshow: int = 20,
) -> OptimizationRecord:

    # Record values for analysis
    parameters_record: dict[str, list[Tensor]] = {
        n: [] for n, _ in optics.named_parameters()
    }
    loss_record = torch.zeros(num_iter)

    default_input = tlm.default_input(sampling)

    show_every = math.ceil(num_iter / nshow)

    for i in range(num_iter):
        optimizer.zero_grad()

        # Evaluate the model
        outputs = optics(default_input)
        loss = outputs.loss

        if not loss.requires_grad:
            raise RuntimeError(
                "No differentiable loss computed by optical stack (loss.requires_grad is False)"
            )

        loss.backward()

        # Record loss values
        loss_record[i] = loss.detach()

        # Record parameter values
        for n, param in optics.named_parameters():
            parameters_record[n].append(param.detach().clone())

        # Compute gradients for gradient magniture
        # and sanity check that gradient isn't nan or inf
        grad = get_all_gradients(optics)
        if torch.isnan(grad).any():
            print("ERROR: nan in grad", grad)
            raise RuntimeError("nan in gradient, check your torch.where() =)")

        optimizer.step()

        if i % show_every == 0 or i == num_iter - 1:
            iter_str = f"[{i+1:>3}/{num_iter}]"
            L_str = f"L= {loss.item():>6.3f} | grad norm= {torch.linalg.norm(grad)}"
            print(f"{iter_str} {L_str}")

    return OptimizationRecord(num_iter, parameters_record, loss_record, optics)


def plot_optimization_record(record: OptimizationRecord) -> None:

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
