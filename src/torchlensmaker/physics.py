import torch


def reflection(rays, normals):
    """
    Vector based reflection.

    Args:
        ray: unit vectors of the incident rays, shape (B, 2)
        normal: unit vectors normal to the surface, shape (B, 2)

    Returns:
        vectors of the reflected vector with shape (B, 2)
    """

    dot_product = torch.sum(rays * normals, dim=1, keepdim=True)
    R = rays - 2 * dot_product * normals
    return R / torch.norm(R, dim=1, keepdim=True)


def refraction(ray, normal, n1, n2, critical_angle='drop'):
    """
    Vector based refraction (Snell's law).
    
    The 'critical_angle' argument specifies how incident rays beyond the
    critical angle are handled:
    
        * 'nan': Incident rays beyond the critical angle will refract
          as nan values. The returned tensor always has the same shape as the
          input tensors.

        * 'clamp': Incident rays beyond the critical angle all refract at 90°.
          The returned tensor always has the same shape as the input tensors.

        * 'drop' (default): Incident rays beyond the critical angle will not be
          refracted. The returned tensor doesn't necesarily have the same shape
          as the input tensors.

    Args:
        ray: unit vectors of the incident rays, shape (B, 2)
        normal: unit vectors normal to the surface, shape (B, 2)
        n1: index of refraction of the incident medium (float)
        n2: index of refraction of the refracted medium (float)
        critical_angle: one of 'nan', 'clamp', 'drop' (default: 'nan')
    
    Returns:
        unit vectors of the refracted rays, shape (C, 2)
    """

    # Compute dot product for the batch, aka cosine of the incident angle
    cos_theta_i = torch.sum(ray * -normal, dim=1, keepdim=True)

    # Compute R_perp and R_para, depending on critical angle options
    R_perp = n1/n2 * (ray + cos_theta_i * normal)

    if critical_angle == 'nan':
        R_para = -torch.sqrt(1 - torch.sum(R_perp * R_perp, dim=1, keepdim=True)) * normal
    
    elif critical_angle == 'clamp':
        radicand = torch.clamp(1 - torch.sum(R_perp * R_perp, dim=1, keepdim=True), min=0., max=None)
        R_para = -torch.sqrt(radicand) * normal
    
    elif critical_angle == 'drop':
        radicand = 1 - torch.sum(R_perp * R_perp, dim=1, keepdim=True)
        valid = (radicand >= 0.0).squeeze()
        R_para = -torch.sqrt(radicand[valid, :]) * normal[valid, :]
        R_perp = R_perp[valid, :]
    
    else:
        raise ValueError(f"critical_angle must be one of 'nan', 'clamp', 'drop'. Got {repr(critical_angle)}.")

    # Combine R_perp and R_para and normalize the result
    R = R_perp + R_para
    return R / torch.norm(R, dim=1, keepdim=True)
