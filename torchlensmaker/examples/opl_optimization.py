"""
Step 6 example: OPL-based wavefront optimization.

Optimizes the curvature of a plano-convex lens to minimize OPL variance
(a proxy for spherical aberration / wavefront error) across rays.
"""

import torch

import torchlensmaker as tlm


def build_lens(C):
    return tlm.Sequential(
        tlm.ObjectAtInfinity(20, 0.5),
        tlm.Gap(80),
        tlm.RefractiveSurface(
            tlm.SphereByCurvature(diameter=15, C=C),
            materials=("air", "BK7"),
        ),
        tlm.Gap(5),
        tlm.RefractiveSurface(
            tlm.Plane(display_diameter=15),
            materials=("BK7", "air"),
        ),
        tlm.Gap(80),
        tlm.FocalPoint(),
    )


def opl_loss(model):
    trace = tlm.raytrace(model, 2)
    # Path from source ("0") to rear surface ("4")
    path = tlm.linear_path(trace, "0", "4")
    opl = path.opl()
    valid = trace.nodes["4"].bundle_out.valid
    # OPL variance over valid rays measures wavefront error
    return opl[valid].var()


def main():
    C = tlm.parameter(0.02)
    model = build_lens(C)
    model.set_sampling2d(pupil=20, field=1, wavel=1)

    optimizer = torch.optim.Adam([C], lr=1e-4)

    print("Optimizing lens curvature to minimize OPL variance (wavefront error)")
    print(f"{'Step':>6}  {'OPL variance':>14}  {'C':>10}")

    for step in range(100):
        optimizer.zero_grad()
        loss = opl_loss(model)
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            print(f"{step:>6}  {loss.item():>14.6f}  {C.item():>10.6f}")

    print(f"\nFinal curvature C = {C.item():.6f}")
    print(f"Final OPL variance = {opl_loss(model).item():.6f}")


if __name__ == "__main__":
    main()
