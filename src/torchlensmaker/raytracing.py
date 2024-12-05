import torch
import numpy as np
from math import pi


from torchlensmaker.shapes.common import normed


def ray_point_squared_distance(ray_origin, ray_vector, point):
    """
    Squared distance from a point to rays
    
    Args:
        ray_origin: tensor of shape (N, 2) - origins of the rays
        ray_vector: tensor of shape (N, 2) - direction unit vectors of the rays
        point: tensor of shape (2,) - the point to compute distance to
    
    Returns:
        tensor of shape (N,) - squared distances from the point to each ray
    """
    # Ensure inputs are the correct shape
    assert ray_origin.shape == ray_vector.shape == (ray_origin.shape[0], 2)
    assert point.shape == (2,)
    
    # Compute line coefficients a, b, c for each ray
    a = -ray_vector[:, 1]
    b = ray_vector[:, 0]
    c = ray_vector[:, 1] * ray_origin[:, 0] - ray_vector[:, 0] * ray_origin[:, 1]
    
    # Broadcast point to match the batch size
    point = point.expand(ray_origin.shape[0], 2)
    
    # Compute the squared distance
    numerator = torch.pow((a * point[:, 0] + b * point[:, 1] + c), 2)
    denominator = torch.pow(a, 2) + torch.pow(b, 2)
    
    return numerator / denominator


def rot2d_matrix(theta):
    return np.array( [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]] )


def rot2d(v, theta):
    m = rot2d_matrix(theta)
    return m.dot(v.T).T


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


def rays_to_coefficients(points, directions):
    """
    Convert rays represented by points and direction vectors to coefficients of
    a line in the form ax+by+c=0.
    
    Args:
        points: torch.Tensor (N, 2) - points on the lines directions:
        torch.Tensor (N, 2) - unit direction vectors
    
    Returns:
        torch.Tensor (N, 3) - line coefficients [a, b, c] for ax + by + c = 0
    """
    
    assert points.numel() == directions.numel()

    if points.numel() == 0:
        return torch.tensor([[]])

    # Extract x and y components
    x, y = points[:, 0], points[:, 1]
    dx, dy = directions[:, 0], directions[:, 1]
    
    # Compute and stack coefficients
    a = dy
    b = -dx
    c = -(a * x + b * y)
    return torch.stack([a, b, c], dim=1)


def position_on_ray(rays_origins, rays_vectors, points):
    """
    Compute t such that points = rays_origins + t * rays_vectors.
    Points are assumed to be on the rays
    """

    # Only pitfall here is to not divide by zero
    Px, Py = points[:, 0], points[:, 1]
    Rx, Ry = rays_origins[:, 0], rays_origins[:, 1]
    dx, dy = rays_vectors[:, 0], rays_vectors[:, 1]
    
    t_fromy = (Py - Ry) / dy
    t_fromx = (Px - Rx) / dx

    return torch.where(torch.isclose(dy, torch.tensor(0.0), rtol=1e-2, atol=1e-2),
        t_fromx, t_fromy)
