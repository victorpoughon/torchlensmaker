import torch


def rot2d(v: torch.Tensor, theta: torch.Tensor | float) -> torch.Tensor:
    """
    Rotate vectors v by angles theta
    Works with either v or theta batched
    """

    theta = torch.as_tensor(theta)

    # Store original dimensions
    v_dim = v.dim()

    # Reshape inputs if necessary
    if v.dim() == 1:
        v = v.unsqueeze(0)  # Add batch dimension if single vector
    if theta.dim() == 0:
        theta = theta.unsqueeze(0)  # Add batch dimension if single angle

    # Create rotation matrices
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    R = torch.stack(
        [
            torch.stack([cos_theta, -sin_theta], dim=-1),
            torch.stack([sin_theta, cos_theta], dim=-1),
        ],
        dim=-2,
    )

    # Perform rotation
    v_rotated = torch.matmul(R, v.unsqueeze(-1)).squeeze(-1)

    if v_dim == 1:
        v_rotated = v_rotated.squeeze(0)

    return v_rotated
