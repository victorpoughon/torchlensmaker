import math
import torch
import torch.nn as nn

import numpy as np


def normed(v):
    return v / torch.linalg.norm(v)


def line_coefficients(A, B):
    """
    Compute the coefficients (a, b, c) of the line equation ax + by + c = 0
    passing through points A and B for multiple pairs of points.
    
    Args:
    A: torch.Tensor of shape (N, 2) representing N points (x1, y1)
    B: torch.Tensor of shape (N, 2) representing N points (x2, y2)
    
    Returns:
    torch.Tensor of shape (N, 3) representing N sets of [a, b, c]
    """
    # Ensure inputs are tensors
    A = torch.as_tensor(A)
    B = torch.as_tensor(B)
    
    # Extract x and y coordinates
    x1, y1 = A[:, 0], A[:, 1]
    x2, y2 = B[:, 0], B[:, 1]
    
    # Compute a, b, c
    a = y2 - y1
    b = x1 - x2
    c = x2*y1 - x1*y2
    
    return torch.stack([a, b, c], dim=1)


def mirror_points(A, B):
    "Mirror points A around points B"
    return torch.column_stack([
        2*B[:, 0] - A[:, 0],
        2*B[:, 1] - A[:, 1]
    ])


def newton_iteration(surface, lines, tn):
    """
    One iteration of Newton's method
    """

    # Compute value and derivative
    a, b, c = lines[:, 0], lines[:, 1], lines[:, 2]
    points = surface.evaluate(tn)
    diff = surface.derivative(tn)

    # Compute L and L'
    L = a*points[:, 0] + b*points[:, 1] + c
    Lp = a*diff[:, 0] + b*diff[:, 1]

    # Update solution
    delta = - L / Lp

    # We might devide by zero above if line is parallel to the surface
    # derivative. This will result in inf which will be clamped to the surface
    # domain by intersect_newton, and the iteration should recover from there.

    return tn + delta


def intersect_newton(surface, lines):
    """
    Intersect shape with N lines, using Newton's method

    Args:
        surface: the shape
        lines :: (N, 3): coefficients (a, b, c) of lines ax+by+c = 0

    Returns:
        ts :: (N,) parametric coordinate of each intersection
        A value outside of the shape's domain indicates that there is no solution.
    """

    assert isinstance(lines, torch.Tensor) and lines.dim() == 2

    # Initialize solutions
    tn = surface.newton_init((lines.shape[0],))

    with torch.no_grad():
        for _ in range(20):  # TODO parameters for newton iterations
            tn = newton_iteration(surface, lines, tn)

            # Clamp to the domain
            # A newton iteration step can lead to a value outside the domain
            # if the solution is outside the domain or if it's close to outide
            # Clamp here so that we remain valid while iterations are not completed
            tn = torch.clamp(tn, *surface.domain())

    # One Newton iteration for backwards
    tn = newton_iteration(surface, lines, tn)

    # The solution can now be outside of the domain
    # after the last newton step, which means 'no solution'

    # Verify the solution
    # Even if the solution is within the domain, it does not necessarily
    # guarantee that it's on the line. So we verify solutions that are within
    # the domain, and if they're not on the line, assign an out of domain value.
    within_domain = torch.logical_and(
        tn <= surface.domain()[1], tn >= surface.domain()[0]
    )

    points = surface.evaluate(tn[within_domain])

    a, b, c = lines[within_domain, 0], lines[within_domain, 1], lines[within_domain, 2]
    px, py = points[:, 0], points[:, 1]
    residuals = a * px + b * py + c

    out_of_domain_value = float("inf")  # TODO ask the shape for a out of domain value

    # placeholder tensor to match tensor sizes in the where() below
    placeholder = torch.zeros_like(tn)
    placeholder[within_domain] = torch.where(
        torch.abs(residuals) < 1e-4, tn[within_domain], out_of_domain_value
    )

    tn = torch.where(within_domain, placeholder, tn)

    return tn
