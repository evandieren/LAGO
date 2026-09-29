import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
from botorch.acquisition import PosteriorMean
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from torch import Tensor

from lago.gp import (
    build_value_gp,
    fit_value_gp,
    posterior_mean_hessian,
    update_value_gp,
)
from lago.iteration import (
    execute_iteration,
    propose_iteration,
)
from lago.trust_region import (
    TrustRegionConfig,
    TrustRegionState,
    get_sr1_tr_config,
)


@dataclass
class EvaluationArchive:
    X: Tensor
    Y: Tensor
    gradients: Tensor
    has_gradient: Tensor

    @property
    def n_function_evals(self) -> int:
        return self.X.shape[0]

    @property
    def n_gradient_evals(self) -> int:
        return int(self.has_gradient.sum().item())

    def cost(self, gradient_cost: float) -> float:
        return (
            self.n_function_evals
            + gradient_cost * self.n_gradient_evals
        )

@dataclass
class LAGOState:
    model: SingleTaskGP
    trust_region: TrustRegionState
    tr_config: TrustRegionConfig

    active_X: Tensor
    active_Y: Tensor

    archive: EvaluationArchive

    iteration: int = 0
    low_ei_count: int = 0
    last_local_improvement: float = math.inf

def make_initial_archive(
    X: Tensor,
    Y: Tensor,
) -> EvaluationArchive:
    n, d = X.shape

    return EvaluationArchive(
        X=X.clone(),
        Y=Y.clone(),
        gradients=torch.full(
            (n, d),
            torch.nan,
            dtype=X.dtype,
            device=X.device,
        ),
        has_gradient=torch.zeros(
            n,
            dtype=torch.bool,
            device=X.device,
        ),
    )

def append_evaluation(
    archive: EvaluationArchive,
    x: Tensor,
    y: Tensor,
    grad: Tensor | None = None,
) -> EvaluationArchive:
    d = x.numel()

    if grad is None:
        grad_row = torch.full(
            (1, d),
            torch.nan,
            dtype=x.dtype,
            device=x.device,
        )
        has_gradient = torch.tensor(
            [False],
            dtype=torch.bool,
            device=x.device,
        )
    else:
        grad_row = grad.reshape(1, d)
        has_gradient = torch.tensor(
            [True],
            dtype=torch.bool,
            device=x.device,
        )

    return EvaluationArchive(
        X=torch.cat(
            [archive.X, x.unsqueeze(0)],
            dim=0,
        ),
        Y=torch.cat(
            [archive.Y, y.reshape(1, 1)],
            dim=0,
        ),
        gradients=torch.cat(
            [archive.gradients, grad_row],
            dim=0,
        ),
        has_gradient=torch.cat(
            [archive.has_gradient, has_gradient],
            dim=0,
        ),
    )

# This is the first informed step.
def get_informed_candidate(
    model: SingleTaskGP,
    bounds: Tensor,
    *,
    num_restarts: int = 20,
    raw_samples: int = 512,
) -> Tensor:
    posterior_mean = PosteriorMean(
        model=model,
        maximize=False,
    )

    candidate, _ = optimize_acqf(
        acq_function=posterior_mean,
        bounds=bounds,
        q=1,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        options={"maxiter": 200},
    )

    return candidate.squeeze(0)

def initialize_lago(
    initial_X: Tensor,
    initial_Y: Tensor,
    objective: Callable[[Tensor], Tensor],
    gradient: Callable[[Tensor], Tensor],
    bounds: Tensor,
    *,
    num_restarts: int = 20,
    raw_samples: int = 512,
) -> LAGOState:

    # ---------------------------------------------------------
    # D0: initial value-only observations
    # ---------------------------------------------------------

    archive = make_initial_archive(
        initial_X,
        initial_Y,
    )

    active_X = initial_X.clone()
    active_Y = initial_Y.clone()

    # ---------------------------------------------------------
    # Fit GP on D0
    # ---------------------------------------------------------

    model = build_value_gp(
        active_X,
        active_Y,
    )

    fit_value_gp(model)

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

    tr_config = get_sr1_tr_config(
        lengthscale,
        bounds,
    )
    # ---------------------------------------------------------
    # First informed global point x^I
    # ---------------------------------------------------------

    x_informed = get_informed_candidate(
        model,
        bounds,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
    )

    f_informed = objective(x_informed)

    archive = append_evaluation(
        archive,
        x_informed,
        f_informed,
    )

    active_X = torch.cat(
        [active_X, x_informed.unsqueeze(0)],
        dim=0,
    )

    active_Y = torch.cat(
        [active_Y, f_informed.reshape(1, 1)],
        dim=0,
    )

    # Condition on x^I, but do not refit hyperparameters.
    model = update_value_gp(
        model,
        active_X,
        active_Y,
        refit=False,
    )

    # ---------------------------------------------------------
    # Initialize trust region at best observed point
    # ---------------------------------------------------------

    best_index = active_Y.squeeze(-1).argmin()

    center = active_X[best_index]
    f_center = active_Y[best_index].squeeze()

    grad_center = gradient(center)

    # Record that gradient in the archive.
    archive.gradients[best_index] = grad_center
    archive.has_gradient[best_index] = True
    # Later we should never assume archive indices and active-GP indices
    # coincide, because filtering breaks that relationship.

    hessian = posterior_mean_hessian(model, center)
    trust_region = TrustRegionState(
        center=center,
        radius=tr_config.initial_radius,
        f_center=f_center,
        grad_center=grad_center,
        hessian=hessian,
    )

    return LAGOState(
        model=model,
        trust_region=trust_region,
        tr_config=tr_config,
        active_X=active_X,
        active_Y=active_Y,
        archive=archive,
    )

def run_lago(
    initial_X: Tensor,
    initial_Y: Tensor,
    objective: Callable[[Tensor], Tensor],
    gradient: Callable[[Tensor], Tensor],
    bounds: Tensor,
    *,
    evaluation_budget: float | None = None,
    gradient_cost: float = 1.0,
    max_iterations: int | None = None,
    gamma: float = 1.0,
    nu: float = 0.1,
    epsilon_t: float = 1e-12,
    n_low_ei: int = 5,
    refit_interval: int = 10,
    step_tolerance: float = 1e-7,
    num_restarts: int = 20,
    raw_samples: int = 512,
    verbose: bool = True,
) -> LAGOState:

    if evaluation_budget is None and max_iterations is None:
        raise ValueError(
            "Specify evaluation_budget, max_iterations, or both."
        )

    if gradient_cost < 0:
        raise ValueError(
            "gradient_cost must be nonnegative."
        )

    state = initialize_lago(
        initial_X=initial_X,
        initial_Y=initial_Y,
        objective=objective,
        gradient=gradient,
        bounds=bounds,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
    )

    while True:

        # -----------------------------------------------------
        # Iteration-boundary stopping conditions
        # -----------------------------------------------------

        if (
            evaluation_budget is not None
            and state.archive.cost(gradient_cost)
            >= evaluation_budget
        ):
            break

        if (
            max_iterations is not None
            and state.iteration >= max_iterations
        ):
            break

        if (
            state.low_ei_count >= n_low_ei
            and state.last_local_improvement < epsilon_t
        ):
            break

        next_iteration = state.iteration + 1

        # -----------------------------------------------------
        # Periodic GP refit
        # -----------------------------------------------------

        if next_iteration % refit_interval == 0:
            if verbose:
                print(
                    f"refitting GP at iteration {next_iteration}, "
                    f"n={state.active_X.shape[0]}"
                )
            state.model = update_value_gp(
                state.model,
                state.active_X,
                state.active_Y,
                refit=True,
            )

        # -----------------------------------------------------
        # Propose both local and global candidates
        # -----------------------------------------------------

        proposal = propose_iteration(
            state=state.trust_region,
            model=state.model,
            bounds=bounds,
            config=state.tr_config,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            step_tolerance=step_tolerance,
        )

        # -----------------------------------------------------
        # Complete the selected iteration
        # -----------------------------------------------------

        result = execute_iteration(
            proposal=proposal,
            model=state.model,
            active_X=state.active_X,
            active_Y=state.active_Y,
            objective=objective,
            gradient=gradient,
            config=state.tr_config,
            gamma=gamma,
            nu=nu,
        )

        # Every paid-for evaluation goes in the archive.
        state.archive = append_evaluation(
            state.archive,
            result.point,
            result.value,
            result.gradient,
        )

        state.model = result.model
        state.trust_region = result.state
        state.active_X = result.active_X
        state.active_Y = result.active_Y
        state.iteration = next_iteration

        # -----------------------------------------------------
        # Update diagnostics used at the NEXT iteration boundary
        # -----------------------------------------------------

        ei = proposal.log_ei.exp().item()
        local_improvement = proposal.local_improvement.item()

        if ei < epsilon_t:
            state.low_ei_count += 1
        else:
            state.low_ei_count = 0

        state.last_local_improvement = local_improvement

        if verbose:
            best_f = state.archive.Y.min().item()

            if result.branch == "local":
                print(
                    f"[{next_iteration:3d}] "
                    f"{result.branch:6s} "
                    f"accepted={result.accepted!s:5s} "
                    f"f={result.value.item():.6e} "
                    f"best={best_f:.6e} "
                    f"EI={proposal.log_ei.exp().item():.3e} "
                    f"I_local={proposal.local_improvement.item():.3e} "
                    f"radius={result.state.radius:.3e}"
                )
            else:
                print(
                    f"[{next_iteration:3d}] "
                    f"{result.branch:6s} "
                    f"f={result.value.item():.6e} "
                    f"best={best_f:.6e} "
                    f"EI={proposal.log_ei.exp().item():.3e} "
                    f"I_local={proposal.local_improvement.item():.3e} "
                    f"radius={result.state.radius:.3e}"
                )



    return state
