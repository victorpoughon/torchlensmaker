import torch
import numpy as np
from math import pi


from torchlensmaker.shapes.common import normed


def normal_vector(A, B):
    """
    Normal vector to (A,B) segment line.
    Returns the one that's in the direction that's 90° rotated from the AB vector.
    The other normal is the opposite.
    """
    # Calculate the direction vector from A to B
    direction = B - A
    
    # Rotate the direction vector by 90 degrees to get the normal vector
    normal = torch.stack([-direction[..., 1], direction[..., 0]], dim=-1)
    
    # Normalize
    normal = normal / torch.norm(normal, dim=-1, keepdim=True)
    
    return normal


def signed_angle(a, b):
    "Signed angle between 2D vectors a and b"

    Y = torch.det(torch.stack([a, b], dim=0))
    X = torch.dot(a, b)
    return torch.arctan2(Y, X)


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


def rot2d_torch(v, theta):
    """
    Rotate 2D vector(s) v by angle theta (in radians).
    
    Args:
    v: torch.Tensor of shape (2,) or (N, 2) where N is the number of vectors
    theta: rotation angle in radians (scalar or torch.Tensor)
    
    Returns:
    Rotated vector(s) with the same shape as input v
    """
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    
    rotation_matrix = torch.tensor([
        [cos_theta, -sin_theta],
        [sin_theta, cos_theta]
    ], dtype=v.dtype, device=v.device)
    
    # Handle both single vector and batch of vectors
    if v.dim() == 1:
        return rotation_matrix @ v
    else:
        return v @ rotation_matrix.T


def refraction(ray, normal, n1, n2):
    """
    Vector based Snell's law

    ray: unit vector of the incident ray
    normal: unit vector normal to the surface
    n1, n2: indices of refraction
    
    Returns: unit vector of the refracted ray
    """

    R_perp = n1/n2 * (ray + (-ray.dot(normal))*normal)
    R_para = -torch.sqrt(1- R_perp.dot(R_perp)) * normal
    return normed(R_perp + R_para)


def clamped_refraction(ray, normal, n1, n2):
    """
    Like refraction, but clamps the incident angle at the critical angle

    ray: unit vector of the incident ray
    normal: unit vector normal to the surface
    n1, n2: indices of refraction
    
    Returns: unit vector of the refracted ray
    """

    R_perp = n1/n2 * (ray + (-ray.dot(normal))*normal)
    radicand = torch.clamp(1- R_perp.dot(R_perp), min=0.0, max=None)
    R_para = -torch.sqrt(radicand) * normal
    return normed(R_perp + R_para)


def super_refraction(incident_ray, normal, n1, n2):
    """
    Continuous extension of Snell's law to allow optimization
    beyond the critical angle.

    ray: unit vector of the incident ray
    normal: unit vector normal to the surface
    n1, n2: indices of refraction
    
    Returns: (v, fake_ray)
    v: unit vector of the refracted ray
    fake_ray: None if the incident angle is less than the critical angle
              The fake
    """

    # Compute angles
    # For numerical stability clamp to [-1, 1]
    # sometimes, the two input vectors are not exactly unit length,
    # so cosine can go slightly over 1.0 if they are well aligned
    cos_theta_i = -torch.dot(incident_ray, normal)
    cos_theta_i = torch.clamp(cos_theta_i, -1.0, 1.0)
    
    sin_theta_i = torch.sqrt(1 - cos_theta_i**2)
    sin_theta_i = torch.clamp(sin_theta_i, -1.0, 1.0)

    fake_R = None

    if n1/n2 * sin_theta_i < 1.0:
        # Normal refraction
        # TODO optim possible here by not recomputing cos_theta_i in refract
        R_refracted = refraction(incident_ray, normal, n1, n2)
    else:
        # Signed incident angle
        theta_i = signed_angle(-normal, incident_ray)
        assert(torch.allclose(torch.abs(theta_i), torch.arcsin(sin_theta_i)))
        
        # Super refraction
        # Rotate the normal to make the fake incident ray
        C = np.arcsin(n2/n1)
        
        super_refract_forward_rotation = pi + torch.arccos(cos_theta_i) - 2*C
        super_refract_rotation = torch.where(theta_i > 0,
            super_refract_forward_rotation, -super_refract_forward_rotation)

        fake_R = rot2d_torch(-normal, super_refract_rotation)
        
        # Flip the normal vector and use fake_R for super refraction
        R_refracted = refraction(fake_R, -normal, n1, n2)

    return R_refracted, fake_R


def rays_to_coefficients(points, directions):
    """
    Convert rays represented by points and direction vectors to coefficients of a line.
    
    Args:
        points: torch.Tensor (N, 2) - points on the lines
        directions: torch.Tensor (N, 2) - unit direction vectors
    
    Returns:
        torch.Tensor (N, 3) - line coefficients [a, b, c] for ax + by + c = 0
    """
    
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
    Compute t such that points = rays_origins + t rays_vectors
    points are assumed to be on the rays
    """

    # Only pitfall here is to not divide by zero
    Px, Py = points[:, 0], points[:, 1]
    Rx, Ry = rays_origins[:, 0], rays_origins[:, 1]
    dx, dy = rays_vectors[:, 0], rays_vectors[:, 1]
    
    t_fromy = (Py - Ry) / dy
    t_fromx = (Px - Rx) / dx

    return torch.where(torch.isclose(dy, torch.tensor(0.0), rtol=1e-2, atol=1e-2),
        t_fromx, t_fromy)
