import argparse
import functools

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes

import torchimplicit as ti

implicit_functions = {
    "sphere": ti.implicit_sphere_3d,
    "disk": ti.implicit_disk_3d,
    "yzcircle": ti.implicit_yzcircle_3d,
}

function_params = {
    "sphere": ["R"],
    "disk": ["R"],
    "yzcircle": ["R"],
}


# signed_field[name](XX_np, YY_np, ZZ_np, params) → ndarray with negative inside, positive outside
def _signed_sphere(XX, YY, ZZ, params):
    R = params[0]
    return np.sqrt(XX**2 + YY**2 + ZZ**2) - R


def _signed_disk(XX, YY, ZZ, params):
    R = params[0]
    r_yz = np.sqrt(YY**2 + ZZ**2)
    within_cylinder = r_yz <= R
    d_plane = XX  # signed: negative for x<0 side, positive for x>0 side — use abs for unsigned dist
    d_circle = np.sqrt(XX**2 + (r_yz - R) ** 2)
    # unsigned distance to disk surface, then we make it negative "inside" the thin slab
    F = np.where(within_cylinder, np.abs(XX), d_circle)
    # Flip sign to get negative close to the surface (F=0 is the surface boundary)
    # This doesn't have a natural "inside" in 3D, so we use -F to invert: surface is where F=0
    return F


def _signed_yzcircle(XX, YY, ZZ, params):
    R = params[0]
    r_yz = np.sqrt(YY**2 + ZZ**2)
    return np.sqrt((r_yz - R) ** 2 + XX**2)


signed_fields = {
    "sphere": _signed_sphere,
    "disk": _signed_disk,
    "yzcircle": _signed_yzcircle,
}

marching_cubes_level = {
    "sphere": 0.0,
    "disk": None,  # computed from data
    "yzcircle": None,  # computed from data
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="3D surface plot for a 3D implicit function in torchimplicit"
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
    parser.add_argument(
        "--N",
        type=int,
        default=50,
        metavar="N",
        help="Grid resolution per axis (default: 100)",
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
    params = torch.tensor(user_params)
    return functools.partial(fn, params=params)


def main():
    args = parse_args()
    fn = bind_function(args.function, args.params)

    R = args.params[0] if args.params else 10.0
    extent = 1.5 * R
    N = args.N

    x = torch.linspace(-extent, extent, N)
    y = torch.linspace(-extent, extent, N)
    z = torch.linspace(-extent, extent, N)
    XX, YY, ZZ = torch.meshgrid(x, y, z, indexing="ij")  # each (N, N, N)

    points = torch.stack([XX.flatten(), YY.flatten(), ZZ.flatten()], dim=-1)

    with torch.no_grad():
        result = fn(points, order=0)

    XX_np, YY_np, ZZ_np = XX.numpy(), YY.numpy(), ZZ.numpy()

    # Compute signed field for marching cubes (negative inside / near surface, positive outside)
    signed_F = signed_fields[args.function](XX_np, YY_np, ZZ_np, args.params or [R])
    signed_F = np.nan_to_num(
        signed_F, nan=float("inf"), posinf=float("inf"), neginf=float("inf")
    )

    fn_name = implicit_functions[args.function].__name__
    param_str = ", ".join(
        f"{k}={v}" for k, v in zip(function_params[args.function], args.params)
    )
    title = f"{fn_name}({param_str})"

    # Extract iso-surface at the zero level. spacing converts voxel indices to world coords.
    spacing = (2 * extent / (N - 1),) * 3
    level = marching_cubes_level[args.function]
    if level is None:
        level = float(np.percentile(np.abs(signed_F), 1))
    verts, faces, normals, _ = marching_cubes(signed_F, level=level, spacing=spacing)

    # Shift vertices so the grid is centred at the origin
    verts -= extent

    plt.style.use("dark_background")

    fig = plt.figure(figsize=(8, 7))
    fig.suptitle(title)
    ax = fig.add_subplot(111, projection="3d")

    mesh = Poly3DCollection(verts[faces], alpha=0.85, linewidth=0)
    mesh.set_facecolor("cyan")
    mesh.set_edgecolor("none")
    ax.add_collection3d(mesh)

    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_zlim(-extent, extent)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_box_aspect([1, 1, 1])

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
