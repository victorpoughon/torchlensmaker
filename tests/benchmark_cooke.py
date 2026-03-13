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

import argparse
import math
import time

import torch

import torchlensmaker as tlm


def _normalize_device(device_str: str) -> torch.device:
    device_str = device_str.lower()
    if device_str == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    if device_str == "cpu":
        return torch.device("cpu")
    return torch.device(device_str)


def _normalize_dtype(dtype_str: str) -> torch.dtype:
    dtype_str = dtype_str.lower()
    if dtype_str == "float32" or dtype_str == "fp32":
        return torch.float32
    if dtype_str == "float64" or dtype_str == "fp64":
        return torch.float64
    raise ValueError(f"Unsupported dtype: {dtype_str}")


class ImagingModel(torch.nn.Module):
    def __init__(self, optics, image_plane):
        super().__init__()
        self.optics = optics
        self.image_plane = image_plane

    def forward(self, data):
        outputs = self.optics(data)
        _, loss = self.image_plane(outputs.rays, outputs.fk, outputs.direction)
        return loss


def optimize(
    optics: tlm.BaseModule,
    optimizer: torch.optim.Optimizer,
    num_iter: int,
    dtype: torch.dtype | None = None,
    nshow: int = 20,
    dim: int = 2,
) -> None:
    if dtype is None:
        dtype = torch.get_default_dtype()

    default_input = tlm.default_input(dim, dtype)

    # We assume the last element is imageplane
    source, core, image_plane = optics[0], optics[1:-1], optics[-1]
    model = ImagingModel(core, image_plane)

    input_rays = source.sequential(default_input)

    print("Compiling model")
    model = torch.compile(model)

    print("Warm up forward")
    model.zero_grad()
    loss = model(input_rays)

    print("Warm up backwards")
    loss.backward()

    print("Start of optimization loop")
    start_time = time.time()

    for i in range(num_iter):
        optimizer.zero_grad()

        # Evaluate the model
        loss = model(input_rays)

        if not loss.requires_grad:
            raise RuntimeError(
                "No differentiable loss computed by optical stack (loss.requires_grad is False)"
            )

        loss.backward()
        optimizer.step()

        iter_str = f"[{i + 1:>3}/{num_iter}]"
        print(f"{iter_str}")

    end_time = time.time()
    duration = end_time - start_time
    print(f"Optimization loop done in {duration:.2f}s")


def benchmark(N: int, num_iter: int, dtype: torch.dtype, device: torch.device) -> None:
    torch.set_default_dtype(dtype)
    torch.set_default_device(device)

    d1, d2 = 30, 25

    r1 = tlm.SphereByRadius(d1, 26.4)
    r2 = tlm.SphereByRadius(d1, -150.7)
    r3 = tlm.SphereByRadius(d2, -29.8)
    r4 = tlm.SphereByRadius(d2, 24.2)
    r5 = tlm.SphereByRadius(d1, 150.7)
    r6 = tlm.SphereByRadius(d1, -26.4)

    material1 = tlm.NonDispersiveMaterial(1.5108)
    material2 = tlm.NonDispersiveMaterial(1.6042)

    L1 = tlm.lenses.singlet(r1, tlm.InnerGap(5.9), r2, material=material1)
    L2 = tlm.lenses.singlet(r3, tlm.InnerGap(0.2), r4, material=material2)
    L3 = tlm.lenses.singlet(r5, tlm.InnerGap(5.9), r6, material=material1)

    focal_gap = tlm.Gap(85, trainable=True)

    optics = tlm.Sequential(
        tlm.ObjectAtInfinity(10, 15),
        L1,
        tlm.Gap(10.9),
        L2,
        tlm.Gap(3.1),
        tlm.Aperture(18),
        tlm.Gap(9.4),
        L3,
        focal_gap,
        tlm.ImagePlane(65),
    )

    sq3 = math.ceil(math.pow(N, 1 / 3))
    # optics.set_sampling3d(pupil=sq3, field=sq3, wavel=sq3)

    optics.set_sampling2d(pupil=sq3, field=sq3, wavel=sq3)

    optimize(
        optics, tlm.optim.Adam(optics.parameters(), lr=5e-4), dim=2, num_iter=num_iter
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("N", type=int, help="number of rays")
    parser.add_argument("num_iter", type=int, help="number of iteration")
    parser.add_argument("--device", "-d", help="device")
    parser.add_argument(
        "--dtype",
        "-t",
        choices=["float32", "float64"],
        help="Data type to use (default: float32)",
    )
    args = parser.parse_args()

    benchmark(
        args.N,
        args.num_iter,
        _normalize_dtype(args.dtype),
        _normalize_device(args.device),
    )


if __name__ == "__main__":
    main()
