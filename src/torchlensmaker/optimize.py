import torch
import torch.nn as nn
import torch.optim as optim

import math

import torchlensmaker as tlm

Tensor = torch.Tensor


def get_all_gradients(model: nn.Module) -> Tensor:
    grads = []
    for param in model.parameters():
        if param.grad is not None:
            grads.append(param.grad.view(-1))
    return torch.cat(grads)


def train(
    optics: nn.Module, dim: int, dtype: torch.dtype, num_iter: int, nshow: int = 20
) -> None:
    optimizer = optim.Adam(optics.parameters(), lr=4e-3)
    sampling = {"dim": dim, "dtype": dtype, "base": 10}

    default_input = tlm.default_input(sampling)

    show_every = math.ceil(num_iter / nshow)

    for i in range(num_iter):
        optimizer.zero_grad()

        # evaluate the model
        outputs = optics(default_input)
        loss = outputs.loss
        loss.backward()

        grad = get_all_gradients(optics)
        if torch.isnan(grad).any():
            print("ERROR: nan in grad", grad)
            raise RuntimeError("nan in gradient, check your torch.where() =)")

        optimizer.step()

        if i % show_every == 0:
            iter_str = f"[{i:>3}/{num_iter}]"
            L_str = f"L= {loss.item():>6.3f} | grad norm= {torch.linalg.norm(grad)}"
            print(f"{iter_str} {L_str}")
