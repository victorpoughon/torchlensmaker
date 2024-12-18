import matplotlib.pyplot as plt
import torch

import torchlensmaker as tlm


from torchlensmaker.optics import linear_magnification


def plot_magnification(optics, sampling):
    """
    Compute and plot magnification data for the given optical system
    The system must compute object and image coordinates
    """

    # TODO add color_dim

    # Evaluate the optical stack
    output = optics(tlm.default_input, sampling)

    # Extract object and image coordinate (called T and V)
    T = output.rays.get("object")
    V = output.rays.get("image")

    mag, _ = linear_magnification(T, V)

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

    plt.show()
