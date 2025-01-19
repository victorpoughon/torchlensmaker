import matplotlib.pyplot as plt
import torch

from torchlensmaker.shapes.common import mirror_points


def render_lines(ax, lines, xlim):
    """
    Render lines on a matplotlib axis.

    Parameters:
    ax (matplotlib.axes.Axes): The matplotlib axis to draw on.
    lines (torch.Tensor): A tensor of shape (N, 3) where each row represents
                          the coefficients [a, b, c] of a line ax + by + c = 0.
    """
    lines = torch.as_tensor(lines)

    # Create x values for plotting
    x = torch.linspace(xlim[0], xlim[1], 100)

    # Iterate through each line
    for line in lines:
        a, b, c = line

        # Handle vertical lines (avoid division by zero)
        if b.abs() < 1e-6:
            if a.abs() < 1e-6:
                continue  # Skip if both a and b are zero
            x_intercept = -c / a
            ax.axvline(x=x_intercept.item(), color="orange")
        else:
            # Calculate y values: y = (-ax - c) / b
            y = (-a * x - c) / b

            # Convert to numpy for plotting
            x_np = x.numpy()
            y_np = y.numpy()

            # Plot the line
            ax.plot(x_np, y_np, color="orange")


def render_spline(ax, spline):
    """
    Render a Bezier Spline shape on a matplotlib axis.
    """

    X, Y, CX, CY = spline.coefficients()
    ax.scatter(X.detach().numpy(), Y.detach().numpy(), color="steelblue", marker="x")
    ax.scatter(CX.detach().numpy(), CY.detach().numpy(), color="#999999", marker=7)

    next_cp = torch.stack((CX, CY), dim=-1)
    next_knot = torch.stack((X, Y), dim=-1)
    mirrors = mirror_points(next_cp, next_knot)
    ax.scatter(
        mirrors[:, 0].detach().numpy(),
        mirrors[:, 1].detach().numpy(),
        color="#999999",
        marker=6,
    )

    sampleTs = torch.linspace(-spline.num_intervals, spline.num_intervals, 50)
    points = spline.evaluate(sampleTs).detach().numpy()
    ax.plot(points[:, 0], points[:, 1], color="steelblue")

    normalst = torch.linspace(-spline.num_intervals, spline.num_intervals, 11)
    normals_origins = spline.evaluate(normalst).detach().numpy()
    normals_vectors = spline.normal(normalst).detach().numpy()
    for o, n in zip(normals_origins, normals_vectors):
        ax.plot([o[0], o[0] + n[0]], [o[1], o[1] + n[1]], color="grey", linestyle="--")

    ax.set_aspect("equal")


def render_collision_points(ax, points):
    """
    Render collision points on a matplotlib axis
    """
    
    if isinstance(points, torch.Tensor):
        points = points.detach().numpy()
    ax.scatter(points[:, 0], points[:, 1], color="green", marker="o")
