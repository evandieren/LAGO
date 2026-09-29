import torch
from botorch.acquisition import LogExpectedImprovement
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples
from torch import Tensor


def get_global_candidate(
    model: SingleTaskGP,
    bounds: Tensor,
    best_f: Tensor | float,
    center: Tensor,
    radius: float,
    *,
    num_restarts: int = 20,
    raw_samples: int = 512,
) -> tuple[Tensor, Tensor]:
    """Maximize LogEI outside the current trust region."""

    log_ei = LogExpectedImprovement(
        model=model,
        best_f=best_f,
        maximize=False,
    )

    def outside_tr(x: Tensor) -> Tensor:
        step = x - center
        return step @ step - radius**2

    samples = draw_sobol_samples(
        bounds=bounds,
        n=raw_samples,
        q=1,
    )

    points = samples[:, 0, :]

    feasible = (
        torch.sum(
            (points - center) ** 2,
            dim=-1,
        )
        >= radius**2
    )

    feasible_samples = samples[feasible]

    if feasible_samples.shape[0] < num_restarts:
        raise RuntimeError(
            "Not enough feasible initial points outside "
            "the trust region."
        )

    # Use the best feasible Sobol points as restarts.
    with torch.no_grad():
        values = log_ei(feasible_samples)

    indices = torch.topk(
        values,
        k=num_restarts,
    ).indices

    initial_conditions = feasible_samples[indices]

    candidate, log_ei_value = optimize_acqf(
        acq_function=log_ei,
        bounds=bounds,
        q=1,
        num_restarts=num_restarts,
        raw_samples=None,
        batch_initial_conditions=initial_conditions,
        nonlinear_inequality_constraints=[
            (outside_tr, True),
        ],
        options={
            "batch_limit": 1,
            "maxiter": 200,
        },
    )

    return (
        candidate.squeeze(0),
        log_ei_value.squeeze(),
    )