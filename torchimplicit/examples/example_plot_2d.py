import argparse
import functools

import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import torch

import torchimplicit as ti

implicit_functions = {
    "disk": ti.implicit_disk_2d,
    "yzcircle": ti.implicit_yzcircle_2d,
    "yaxis": ti.implicit_yaxis_2d,
    "sphere": ti.implicit_sphere_2d,
}

function_params = {
    "disk": ["R"],
    "yzcircle": ["R"],
    "yaxis": [],
    "sphere": ["R"],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Example plot for a 2D implicit function in torchimplicit"
    )
    parser.add_argument(
        "function", choices=list(implicit_functions.keys()), help="Function name"
    )
    parser.add_argument(
        "params", nargs="*", type=float, help="Function parameters (e.g. R)"
    )
    parser.add_argument(
        "--output",
        "-o",
        metavar="FILE",
        help="Save to file instead of showing interactively",
    )
    return parser.parse_args()


def bind_function(name, user_params):
    param_names = function_params[name]
    if len(user_params) != len(param_names):
        raise ValueError(
            f"'{name}' expects {len(param_names)} parameter(s)"
            + (f" ({', '.join(param_names)})" if param_names else "")
            + f", got {len(user_params)}"
        )
    fn = implicit_functions[name]
    kwargs = dict(zip(param_names, user_params))
    return functools.partial(fn, **kwargs) if kwargs else fn


def main():
    args = parse_args()
    fn = bind_function(args.function, args.params)

    R = args.params[0] if args.params else 10.0
    extent = 1.5 * R
    N = 600

    x = torch.linspace(-extent, extent, N)
    y = torch.linspace(-extent, extent, N)
    XX, YY = torch.meshgrid(x, y, indexing="xy")  # XX[i,j]=x[j], YY[i,j]=y[i]

    points = torch.stack([XX.flatten(), YY.flatten()], dim=-1)

    with torch.no_grad():
        result = fn(points, order=1)

    # F[i,j] = f(x[j], y[i]) — matches pcolormesh(x, y, F) convention
    F = result.val.reshape(N, N).numpy()
    F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)

    G = result.grad.reshape(N, N, 2).numpy()
    G = np.nan_to_num(G, nan=0.0, posinf=0.0, neginf=0.0)
    GX, GY = G[..., 0], G[..., 1]
    grad_mag = np.sqrt(GX**2 + GY**2)

    x_np = x.numpy()
    y_np = y.numpy()
    XX_np = XX.numpy()
    YY_np = YY.numpy()

    fn_name = implicit_functions[args.function].__name__
    param_str = ", ".join(
        f"{k}={v}" for k, v in zip(function_params[args.function], args.params)
    )
    title = f"{fn_name}({param_str})"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(title)

    step = N // 20
    GX_norm = GX / (grad_mag + 1e-10)
    GY_norm = GY / (grad_mag + 1e-10)

    # Left: function value with symlog colormap + gradient direction
    vmax = float(F.max())
    norm = matplotlib.colors.SymLogNorm(linthresh=R * 0.01, vmin=-vmax, vmax=vmax)
    ax1.pcolormesh(x_np, y_np, F, norm=norm, cmap="RdBu_r", shading="auto")

    ax1.set_title("f(x, y)")
    ax1.set_aspect("equal")

    # Right: gradient magnitude + gradient direction
    vmax_grad = float(np.percentile(grad_mag, 99)) * 1.5
    ax2.pcolormesh(
        x_np, y_np, grad_mag, cmap="viridis", shading="auto", vmin=0, vmax=vmax_grad
    )
    ax2.quiver(
        XX_np[::step, ::step],
        YY_np[::step, ::step],
        GX_norm[::step, ::step],
        GY_norm[::step, ::step],
        color="k",
        alpha=0.5,
        pivot="mid",
        scale=35,
    )
    ax2.set_title("|∇f(x, y)|")
    ax2.set_aspect("equal")

    plt.tight_layout()

    if args.output:
        plt.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"Saved to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
