from dataclasses import dataclass

import torch
from torch import Tensor

from ._trust_region_solver import (
    solve_box_trust_region_subproblem,
    solve_trust_region_subproblem,
)


@dataclass(frozen=True)
class TrustRegionConfig:
    initial_radius: float
    acceptance_threshold: float
    shrink_threshold: float
    expand_threshold: float
    shrink_factor: float
    expand_factor: float
    max_radius: float

def get_sr1_tr_config(
    lengthscale: float,
    bounds: Tensor,
) -> TrustRegionConfig:
    diameter = torch.linalg.norm(
        bounds[1] - bounds[0]
    ).item()

    return TrustRegionConfig(
        initial_radius=min(
            lengthscale / 2.0,
            diameter / 8.0,
        ),
        acceptance_threshold=5e-4,
        shrink_threshold=0.25,
        expand_threshold=0.75,
        shrink_factor=0.5,
        expand_factor=2.0,
        max_radius=diameter / 2.0,
    )

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

def is_box_feasible(
    point: Tensor,
    lower_bounds: Tensor,
    upper_bounds: Tensor,
) -> bool:
    return bool(
        torch.all(point >= lower_bounds)
        and torch.all(point <= upper_bounds)
    )

def is_tr_step_feasible(
    step: Tensor,
    state: TrustRegionState,
    lower_bounds: Tensor,
    upper_bounds: Tensor,
    tolerance: float = 1e-8,
) -> bool:
    """Check finiteness, box feasibility, and TR feasibility."""

    if not torch.all(torch.isfinite(step)).item():
        return False

    candidate = state.center + step

    if not is_box_feasible(
        candidate,
        lower_bounds - tolerance,
        upper_bounds + tolerance,
    ):
        return False

    step_norm = torch.linalg.norm(step).item()

    return step_norm <= state.radius * (1 + tolerance)

def project_onto_tr_ball(
    step: Tensor,
    radius: float,
) -> Tensor:
    """Radially project a step onto the trust-region ball."""

    step_norm = torch.linalg.norm(step)

    if step_norm <= radius or step_norm == 0:
        return step.clone()

    return (radius / step_norm) * step

def get_box_constrained_candidate(
    state: TrustRegionState,
    lower_bounds: Tensor,
    upper_bounds: Tensor,
    feasibility_tolerance: float = 1e-8,
) -> Tensor:
    """
    Get a local candidate satisfying box and TR constraints.
    
    Box candidate
      │
      ├─ feasible? → return
      │
      └─ otherwise
            │
            ├─ radial projection if possible
            ├─ Cauchy candidate if possible
            │
            └─ choose best model value
                 or zero if none feasible
    """

    step_lower = lower_bounds - state.center
    step_upper = upper_bounds - state.center

    result = solve_box_trust_region_subproblem(
        grad=state.grad_center,
        hessian=state.hessian,
        radius=state.radius,
        step_lower=step_lower,
        step_upper=step_upper,
    )

    solver_step = result.step

    solver_value = quadratic_model(
        state.f_center,
        state.grad_center,
        state.hessian,
        solver_step,
    )

    solver_valid = (
        is_tr_step_feasible(
            solver_step,
            state,
            lower_bounds,
            upper_bounds,
            feasibility_tolerance,
        )
        and torch.isfinite(solver_value).item()
    )

    if solver_valid:
        return solver_step

    candidates: list[Tensor] = []

    # Match Julia: if the numerical candidate respects the box,
    # radially project it onto the TR ball and retain it if feasible.
    box_feasible = (
        torch.all(torch.isfinite(solver_step)).item()
        and torch.all(
            solver_step
            >= step_lower - feasibility_tolerance
        ).item()
        and torch.all(
            solver_step
            <= step_upper + feasibility_tolerance
        ).item()
    )

    if box_feasible:
        projected = project_onto_tr_ball(
            solver_step,
            state.radius,
        )

        if is_tr_step_feasible(
            projected,
            state,
            lower_bounds,
            upper_bounds,
            feasibility_tolerance,
        ):
            candidates.append(projected)

    # Nocedal-Wright Cauchy backup.
    cauchy = cauchy_point(
        state.grad_center,
        state.hessian,
        state.radius,
    )

    if is_tr_step_feasible(
        cauchy,
        state,
        lower_bounds,
        upper_bounds,
        feasibility_tolerance,
    ):
        candidates.append(cauchy)

    if not candidates:
        return torch.zeros_like(state.center)

    values = [
        quadratic_model(
            state.f_center,
            state.grad_center,
            state.hessian,
            step,
        )
        for step in candidates
    ]

    best_index = int(
        torch.argmin(torch.stack(values)).item()
    )

    return candidates[best_index]

def get_local_candidate(
    state: TrustRegionState,
    lower_bounds: Tensor,
    upper_bounds: Tensor,
    feasibility_tolerance: float = 1e-8,
) -> Tensor:
    """
    Compute the local LAGO trust-region candidate.
    
    Arguments:
    - state : current trust-region state.
    - lower_bounds : lower bounds for the box constraints.
    - upper_bounds : upper bounds for the box constraints.
    - feasibility_tolerance : tolerance for feasibility checks.

    Returns:
    - step : local candidate step.
    """

    # Trying the unconstrained trust-region solver first.
    result = solve_trust_region_subproblem(
        grad=state.grad_center,
        hessian=state.hessian,
        radius=state.radius,
    )

    step = result.step

    model_value = quadratic_model(
        state.f_center,
        state.grad_center,
        state.hessian,
        step,
    )

    # checking if valid
    ball_solution_valid = (
        is_tr_step_feasible(
            step,
            state,
            lower_bounds,
            upper_bounds,
            feasibility_tolerance,
        )
        and torch.isfinite(model_value).item()
    )

    if ball_solution_valid:
        return step

    # If the unconstrained solver fails, we fall back to the box-constrained solver.
    return get_box_constrained_candidate(
        state=state,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        feasibility_tolerance=feasibility_tolerance,
    )

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

def reinitialize_trust_region(
    center: Tensor,
    f_center: Tensor,
    grad_center: Tensor,
    hessian: Tensor,
    config: TrustRegionConfig,
) -> TrustRegionState:
    return TrustRegionState(
        center=center,
        radius=config.initial_radius,
        f_center=f_center,
        grad_center=grad_center,
        hessian=hessian,
        terminated=False,
    )