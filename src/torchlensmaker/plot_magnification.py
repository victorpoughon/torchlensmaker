import matplotlib.pyplot as plt
import torch

import torchlensmaker as tlm

def plot_magnification(optics, sampling):
    """
    Compute and plot magnification data for the given optical system
    The system must compute object and image coordinates
    """

    # Evaluate the optical stack
    output = optics(tlm.default_input, sampling)

    # Extract object and image coordinate (called T and V)
    T = output.rays.get("object")
    V = output.rays.get("image")

    # Fit linear magnification and compute residuals
    mag = torch.sum(T * V) / torch.sum(T**2)
    residuals = V - mag * T

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(T.detach().numpy(), V.detach().numpy(), linestyle="none", marker="+")

    X = torch.linspace(T.min().item(), T.max().item(), 50)
    ax.plot(
        X.detach().numpy(),
        (mag * X).detach().numpy(),
        color="lightgrey",
        label=f"mag = {mag:.2f}",
    )

    ax.set_xlabel("Object coordinates")
    ax.set_ylabel("Image coordinates")
    ax.legend()
