import torch
from torch import Tensor

from lago.optimizer import LAGOState


def equivalent_evaluation_costs(
    state: LAGOState,
    gradient_cost: float,
) -> Tensor:
    """
    Return cumulative equivalent function-evaluation cost for each
    function evaluation stored in the archive.

    Cost model:
        cost = n_function_evals + gradient_cost * n_gradient_evals

    Assumes the current LAGO-BO evaluation order:
        - N0 initial function evaluations,
        - one informed function evaluation,
        - one gradient evaluation to initialize the trust region,
        - one function evaluation per main-loop iteration,
        - optional gradient evaluation in an iteration.

    The initialization gradient may be attached in the archive to an
    earlier DOE row, so its cost is charged at the end of initialization
    rather than at that row.
    """
    if gradient_cost < 0:
        raise ValueError("gradient_cost must be nonnegative.")

    archive = state.archive

    n_function_evals = archive.n_function_evals

    n_initial = (
        n_function_evals
        - 1
        - state.iteration
    )

    if n_initial < 1:
        raise ValueError(
            "Could not infer the initial design size from the LAGO state."
        )

    gradient_evals = archive.has_gradient.to(
        dtype=archive.Y.dtype,
        device=archive.Y.device,
    ).clone()

    # Rows [0, ..., n_initial - 1] are the initial DOE.
    # Row n_initial is the informed evaluation.
    informed_index = n_initial

    # Exactly one gradient is paid during initialization, after the
    # informed evaluation and selection of the initial TR centre.
    gradient_evals[: informed_index + 1] = 0.0
    gradient_evals[informed_index] = 1.0

    cost_per_evaluation = (
        torch.ones(
            n_function_evals,
            dtype=archive.Y.dtype,
            device=archive.Y.device,
        )
        + gradient_cost * gradient_evals
    )

    return torch.cumsum(
        cost_per_evaluation,
        dim=0,
    )