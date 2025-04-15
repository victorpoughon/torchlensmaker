import torch


def rays_cylinder_collision(
    P: torch.Tensor,
    V: torch.Tensor,
    xmin: torch.Tensor,
    xmax: torch.Tensor,
    tau: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rays-cylinder collision

    Collision detection between 3D rays and and a cylinder aligned with the X axis

    Args:
        P: rays origins, shape (N, 3)
        V: rays vectors, shape (N, 3)
        xmin: start x coordinate of the cylinder, zero-dim tensor
        xmax: end x coordinate of the cylinder, zero-dim tensor
        tau: radius of the cylinder, zero-dim tensor

    Returns
        t1, t2, hit_mask: tensors of shape (N,)
    """

    # Sanity checks
    assert P.shape == V.shape
    assert P.shape[-1] == V.shape[-1] == 3
    assert xmin.dim() == xmax.dim() == tau.dim() == 0

    # Constants
    const_zero = torch.zeros((1), dtype=P.dtype)

    # Points 0 and 1: Intersection with the cylinder body

    # Quadratic equation coefficients
    A = V[..., 1] ** 2 + V[..., 2] ** 2
    B = 2 * (P[..., 1] * V[..., 1] + P[..., 2] * V[..., 2])
    C = P[..., 1] ** 2 + P[..., 2] ** 2 - tau**2
    delta = B**2 - 4 * A * C
    sqrt_delta = torch.sqrt(torch.clamp(delta, min=const_zero))

    # Ignore A = 0 case, as end cap collision points will take over
    mask_quadratic_solvable = torch.logical_and(delta >= const_zero, A != const_zero)

    # Solve the quadratic
    root0 = (-B + sqrt_delta) / (2 * A)
    root1 = (-B - sqrt_delta) / (2 * A)

    # Finally, check that the quadratic roots are within (xmin, xmax)
    root0_x = P[..., 0] + root0 * V[..., 0]
    root1_x = P[..., 0] + root1 * V[..., 0]
    root0_within_range = torch.logical_and(root0_x >= xmin, root0_x <= xmax)
    root1_within_range = torch.logical_and(root1_x >= xmin, root1_x <= xmax)
    root0_hit_mask = torch.logical_and(mask_quadratic_solvable, root0_within_range)
    root1_hit_mask = torch.logical_and(mask_quadratic_solvable, root1_within_range)

    # Point 2 and 3: Intersection with top and bottom circular caps

    # Solve planar equation, ignore Vx = 0 case, cylinder body collision will take over
    mask_planar_solvable = V[..., 0] != const_zero
    safe_denom = torch.where(mask_planar_solvable, V[..., 0], const_zero)
    root2 = torch.where(
        mask_planar_solvable, (xmin - P[..., 0]) / safe_denom, const_zero
    )
    root3 = torch.where(
        mask_planar_solvable, (xmax - P[..., 0]) / safe_denom, const_zero
    )

    # Check that the planar roots and within radius distance
    root2_r2 = (P[..., 1] + root2 * V[..., 1]) ** 2 + (
        P[..., 2] + root2 * V[..., 2]
    ) ** 2
    root3_r2 = (P[..., 1] + root3 * V[..., 1]) ** 2 + (
        P[..., 2] + root3 * V[..., 2]
    ) ** 2
    root2_within_range = root2_r2 <= tau**2
    root3_within_range = root3_r2 <= tau**2
    root2_hit_mask = torch.logical_and(mask_planar_solvable, root2_within_range)
    root3_hit_mask = torch.logical_and(mask_planar_solvable, root3_within_range)

    # Compute the final hit mask
    stacked_hit_mask = torch.stack(
        (root0_hit_mask, root1_hit_mask, root2_hit_mask, root3_hit_mask), dim=-1
    )
    final_hit_mask = stacked_hit_mask.sum(dim=-1) >= 2

    # Sort stacked hit mask by valid / invalid to get the indices of the first 2 valid roots
    _, indices = torch.sort(-1 * stacked_hit_mask, dim=-1)

    roots = torch.stack((root0, root1, root2, root3), dim=-1)
    good_roots = torch.gather(roots, 1, indices[:, :2])

    return good_roots[:, 0], good_roots[:, 1], final_hit_mask
