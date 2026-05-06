import argparse
import functools

import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import torch

import torchimplicit as ti

registry_2d = {
    k.removesuffix("_2d"): v for k, v in ti.get_implicit_functions(dim=2).items()
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Example plot for a 2D implicit function in torchimplicit"
    )
    parser.add_argument(
        "function", choices=list(registry_2d.keys()), help="Function name"
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
    fdef = registry_2d[name]
    if len(user_params) != fdef.n_params:
        raise ValueError(
            f"'{name}' expects {fdef.n_params} parameter(s)"
            + (f" ({', '.join(fdef.param_names)})" if fdef.param_names else "")
            + f", got {len(user_params)}"
        )
    params = torch.tensor(user_params)
    return functools.partial(fdef.func, params=params)


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
        result = fn(points, order=2)

    # F[i,j] = f(x[j], y[i]) — matches pcolormesh(x, y, F) convention
    F = result.val.reshape(N, N).numpy()
    F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)

    assert result.grad is not None
    G = result.grad.reshape(N, N, 2).numpy()
    G = np.nan_to_num(G, nan=0.0, posinf=0.0, neginf=0.0)
    GX, GY = G[..., 0], G[..., 1]
    grad_mag = np.sqrt(GX**2 + GY**2)

    assert result.hess is not None
    H = result.hess.reshape(N, N, 2, 2).numpy()
    H = np.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0)
    frob = np.sqrt(H[..., 0, 0] ** 2 + 2 * H[..., 0, 1] ** 2 + H[..., 1, 1] ** 2)

    x_np = x.numpy()
    y_np = y.numpy()
    XX_np = XX.numpy()
    YY_np = YY.numpy()

    fdef = registry_2d[args.function]
    param_str = ", ".join(f"{k}={v}" for k, v in zip(fdef.param_names, args.params))
    title = f"{fdef.name}({param_str})"

    plt.style.use("dark_background")

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title)

    step = N // 20
    GX_norm = GX / (grad_mag + 1e-10)
    GY_norm = GY / (grad_mag + 1e-10)

    # Left: plot 1/f so the surface (f=0) glows bright and background fades to black
    cmap_cyan = matplotlib.colors.LinearSegmentedColormap.from_list(
        "black_cyan_white",
        [
            (0.0, "black"),
            (0.95, "cyan"),
            (0.99, "white"),
            (1.0, "white"),
        ],
    )
    inv_F = 1.0 / np.maximum(F, 1e-10)
    vmax_inv = float(np.percentile(inv_F, 99))
    ax1.pcolormesh(
        x_np, y_np, inv_F, cmap=cmap_cyan, shading="auto", vmin=0, vmax=vmax_inv
    )
    ax1.set_title("1 / F")
    ax1.set_aspect("equal")

    # Right: gradient magnitude (plasma) + unit direction arrows
    vmax_grad = float(np.percentile(grad_mag, 99)) * 1.5
    ax2.pcolormesh(
        x_np, y_np, grad_mag, cmap="plasma", shading="auto", vmin=0, vmax=vmax_grad
    )
    ax2.quiver(
        XX_np[::step, ::step],
        YY_np[::step, ::step],
        -GX_norm[::step, ::step],
        -GY_norm[::step, ::step],
        color="k",
        alpha=0.6,
        pivot="tail",
        scale=30,
    )
    ax2.set_title("— ∇ F")
    ax2.set_aspect("equal")

    # Right: Frobenius norm of Hessian — magnitude of total second-order variation
    vmax_frob = float(np.percentile(frob, 99)) * 1.5
    ax3.pcolormesh(
        x_np, y_np, frob, cmap="plasma", shading="auto", vmin=0, vmax=vmax_frob
    )
    ax3.set_title("‖H‖  [Frobenius norm]")
    ax3.set_aspect("equal")

    plt.tight_layout()

    if args.output:
        plt.savefig(
            args.output, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor()
        )
        print(f"Saved to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
