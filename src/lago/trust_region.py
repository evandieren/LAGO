from dataclasses import dataclass

import torch
from torch import Tensor

@dataclass(frozen=True)
class TrustRegionConfig:
    initial_radius: float
    acceptance_threshold: float
    shrink_threshold: float
    expand_threshold: float
    shrink_factor: float
    expand_factor: float
    max_radius: float

@dataclass
class TrustRegionState:
    center : Tensor
    radius : float
    f_center : Tensor
    grad_center : Tensor
    hessian : Tensor
    terminated : bool = False

def quadratic_model(
    f: Tensor,
    grad: Tensor,
    hessian: Tensor,
    step: Tensor,
) -> Tensor:
    return f + grad @ step + 0.5 * step @ hessian @ step

def predicted_improvement(
    f: Tensor,
    grad: Tensor,
    hessian: Tensor,
    step: Tensor,
) -> Tensor:
    return f - quadratic_model(f, grad, hessian, step)

def improvement_ratio(
    f_center: Tensor,
    f_trial: Tensor,
    predicted_decrease: Tensor,
) -> Tensor:
    actual_decrease = f_center - f_trial
    return actual_decrease / predicted_decrease

def sr1_update(
        hessian:Tensor,
        step:Tensor,
        grad_difference:Tensor,
        tolerance: float = 1e-8,
) -> Tensor:
    """
    Apply a safeguarded symmetric rank-one update.

    Arguments:
    - hessian: Hessian of the current quadratic model.
    - step = x_{trial} - x_{k}.
    - grad_difference = grad f(x_{trial}) - grad f(x_{k}).
    - tolerance (default 1e-8) : tolerance for the update, `r` in the paper.

    Returns:
    (Updated) Hessian.
    """

    residual = grad_difference - hessian @ step

    denominator = residual @ step

    threshold = tolerance * torch.linalg.norm(step)*torch.linalg.norm(residual)

    if torch.abs(denominator) <= threshold:
        return hessian

    return hessian + torch.outer(residual, residual) / denominator

def update_radius(
    radius: float,
    rho: float,
    step: Tensor,
    config: TrustRegionConfig,
) -> float:
    """
    Update the trust-region radius.
    
    Arguments:
    - radius : current trust-region radius.
    - rho : improvement ratio.
    - step : step taken in the trust-region subproblem.
    - config : TrustRegionConfig object containing the trust-region parameters.

    Returns:
    - new trust-region radius.
    """
    if rho < config.shrink_threshold:
        # rho is too low, we reduce the radius.
        return config.shrink_factor * radius

    step_norm = torch.linalg.norm(step).item()

    if rho > config.expand_threshold and step_norm > 0.8 * radius:
        # rho is high enough, and we are making a step close to the current TR border.
        # -> we increase the radius.
        return min(config.expand_factor * radius, config.max_radius)

    # either rho not high enough, or step not so close to the current border.
    # -> keep the current radius.
    return radius

# Trust-region subproblem solvers.

def cauchy_point(
    grad: Tensor,
    hessian: Tensor,
    radius: float,
    grad_tol: float = 1e-14,
) -> Tensor:
    """Compute the Cauchy point of a quadratic trust-region model."""

    if radius < 0:
        raise ValueError("Trust-region radius must be nonnegative.")

    grad_norm = torch.linalg.norm(grad)

    if not torch.isfinite(grad_norm):
        raise ValueError("Gradient norm must be finite.")

    if grad_norm <= grad_tol or radius == 0:
        return torch.zeros_like(grad)

    curvature = grad @ hessian @ grad

    if curvature <= 0:
        tau = 1.0
    else:
        tau = min(
            grad_norm**3 / (radius * curvature),
            1.0,
        )

    step = -(tau * radius / grad_norm) * grad

    # Floating-point safeguard.
    step_norm = torch.linalg.norm(step)
    if step_norm > radius:
        step = step * (radius / step_norm)

    return step

def is_box_feasible(
    point: Tensor,
    lower_bounds: Tensor,
    upper_bounds: Tensor,
) -> bool:
    return bool(
        torch.all(point >= lower_bounds)
        and torch.all(point <= upper_bounds)
    )