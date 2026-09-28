from dataclasses import dataclass, replace
from typing import Callable

import torch
from torch import Tensor
from botorch.models import SingleTaskGP

from lago.global_search import get_global_candidate
from lago.gp import (
    posterior_mean_hessian,
    update_value_gp,
)

from lago.trust_region import (
    TrustRegionConfig,
    TrustRegionState,
    improvement_ratio,
    predicted_improvement,
    sr1_update,
    update_radius,
    reinitialize_trust_region,
    get_local_candidate,
)


@dataclass
class LocalStepResult:
    state: TrustRegionState
    active_X: Tensor
    active_Y: Tensor

    trial_point: Tensor
    f_trial: Tensor
    grad_trial: Tensor

    rho: Tensor
    accepted: bool

@dataclass
class IterationProposal:
    state: TrustRegionState
    step: Tensor
    local_improvement: Tensor
    x_global: Tensor
    log_ei: Tensor

@dataclass
class IterationResult:
    state: TrustRegionState
    model: SingleTaskGP
    active_X: Tensor
    active_Y: Tensor

    point: Tensor
    value: Tensor
    gradient: Tensor | None

    branch: str
    accepted: bool | None


def filter_active_data(
    active_X: Tensor,
    active_Y: Tensor,
    center: Tensor,
    f_center: Tensor,
    *,
    lengthscale: float,
    nu: float,
) -> tuple[Tensor, Tensor]:
    """Apply LAGO's lengthscale-based GP filtering rule."""

    distances = torch.linalg.norm(
        active_X - center,
        dim=-1,
    )

    keep = distances > nu * lengthscale

    filtered_X = torch.cat(
        [
            active_X[keep],
            center.unsqueeze(0),
        ],
        dim=0,
    )

    filtered_Y = torch.cat(
        [
            active_Y[keep],
            f_center.reshape(1, 1),
        ],
        dim=0,
    )

    return filtered_X, filtered_Y


def perform_local_step(
    state: TrustRegionState,
    step: Tensor,
    objective: Callable[[Tensor], Tensor],
    gradient: Callable[[Tensor], Tensor],
    active_X: Tensor,
    active_Y: Tensor,
    config: TrustRegionConfig,
    *,
    lengthscale: float,
    nu: float,
) -> LocalStepResult:
    """Evaluate and perform one SR1 trust-region step."""

    trial_point = state.center + step

    f_trial = objective(trial_point)
    grad_trial = gradient(trial_point)

    predicted_decrease = predicted_improvement(
        state.f_center,
        state.grad_center,
        state.hessian,
        step,
    )

    rho = improvement_ratio(
        state.f_center,
        f_trial,
        predicted_decrease,
    )

    new_radius = update_radius(
        radius=state.radius,
        rho=float(rho.item()),
        step=step,
        config=config,
    )

    # SR1 uses the trial gradient even if the trial point is rejected.
    new_hessian = sr1_update(
        hessian=state.hessian,
        step=step,
        grad_difference=grad_trial - state.grad_center,
    )

    accepted = bool(
        rho.item() > config.acceptance_threshold
    )

    if accepted:
        new_state = TrustRegionState(
            center=trial_point,
            radius=new_radius,
            f_center=f_trial,
            grad_center=grad_trial,
            hessian=new_hessian,
        )

        active_X, active_Y = filter_active_data(
            active_X,
            active_Y,
            new_state.center,
            new_state.f_center,
            lengthscale=lengthscale,
            nu=nu,
        )

    else:
        new_state = TrustRegionState(
            center=state.center,
            radius=new_radius,
            f_center=state.f_center,
            grad_center=state.grad_center,
            hessian=new_hessian,
        )

    return LocalStepResult(
        state=new_state,
        active_X=active_X,
        active_Y=active_Y,
        trial_point=trial_point,
        f_trial=f_trial,
        grad_trial=grad_trial,
        rho=rho,
        accepted=accepted,
    )

def propose_iteration(
    state: TrustRegionState,
    model: SingleTaskGP,
    bounds: Tensor,
    config: TrustRegionConfig,
    *,
    num_restarts: int = 20,
    raw_samples: int = 512,
    step_tolerance: float = 1e-7,
) -> IterationProposal:

    lower = bounds[0]
    upper = bounds[1]

    lengthscale = (
        model.covar_module
        .base_kernel
        .lengthscale
        .detach()
        .reshape(-1)
    )

    if lengthscale.numel() != 1:
        raise NotImplementedError(
            "LAGO currently requires a scalar lengthscale."
        )

    lengthscale = lengthscale.item()

    step = get_local_candidate(
        state,
        lower,
        upper,
    )

    local_improvement = predicted_improvement(
        state.f_center,
        state.grad_center,
        state.hessian,
        step,
    )

    step_norm = torch.linalg.norm(step).item()

    terminated = (
        state.terminated
        or step_norm <= step_tolerance
    )

    if terminated:
        state = replace(
            state,
            radius=min(
                state.radius,
                lengthscale / 2.0,
            ),
            terminated=True,
        )

    x_global, log_ei = get_global_candidate(
        model=model,
        bounds=bounds,
        best_f=state.f_center,
        center=state.center,
        radius=state.radius,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
    )

    return IterationProposal(
        state=state,
        step=step,
        local_improvement=local_improvement,
        x_global=x_global,
        log_ei=log_ei,
    )

def execute_iteration(
    proposal: IterationProposal,
    model: SingleTaskGP,
    active_X: Tensor,
    active_Y: Tensor,
    objective: Callable[[Tensor], Tensor],
    gradient: Callable[[Tensor], Tensor],
    config: TrustRegionConfig,
    *,
    gamma: float = 1.0,
    nu: float = 0.1,
) -> IterationResult:

    state = proposal.state
    step = proposal.step
    x_global = proposal.x_global

    lengthscale = (
        model.covar_module
        .base_kernel
        .lengthscale
        .detach()
        .item()
    )

    if state.terminated:
        use_global = True
    else:
        use_global = bool(
            proposal.log_ei.exp()
            > gamma * proposal.local_improvement
        )

    if use_global:
        f_global = objective(x_global)
        grad_global = None

        active_X = torch.cat(
            [active_X, x_global.unsqueeze(0)],
            dim=0,
        )
        active_Y = torch.cat(
            [active_Y, f_global.reshape(1, 1)],
            dim=0,
        )

        model = update_value_gp(
            model,
            active_X,
            active_Y,
        )

        if f_global < state.f_center:
            grad_global = gradient(x_global)

            hessian = posterior_mean_hessian(
                model,
                x_global,
            )

            state = reinitialize_trust_region(
                center=x_global,
                f_center=f_global,
                grad_center=grad_global,
                hessian=hessian,
                config=config,
            )

        return IterationResult(
            state=state,
            model=model,
            active_X=active_X,
            active_Y=active_Y,
            point=x_global,
            value=f_global,
            gradient=grad_global,
            branch="global",
            accepted=None,
        )

    local = perform_local_step(
        state=state,
        step=step,
        objective=objective,
        gradient=gradient,
        active_X=active_X,
        active_Y=active_Y,
        config=config,
        lengthscale=lengthscale,
        nu=nu,
    )

    if local.accepted:
        model = update_value_gp(
            model,
            local.active_X,
            local.active_Y,
        )

    return IterationResult(
        state=local.state,
        model=model,
        active_X=local.active_X,
        active_Y=local.active_Y,
        point=local.trial_point,
        value=local.f_trial,
        gradient=local.grad_trial,
        branch="local",
        accepted=local.accepted,
    )