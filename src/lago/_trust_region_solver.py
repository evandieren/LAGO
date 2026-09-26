from dataclasses import dataclass

import numpy as np
import torch
from scipy.optimize._trustregion_exact import IterativeSubproblem
from torch import Tensor


_K_EASY = 1e-10
_K_HARD = 1e-6
_MAXITER = 100


@dataclass(frozen=True)
class TrustRegionSubproblemResult:
    step: Tensor
    hits_boundary: bool
    multiplier: float
    iterations: int

def solve_trust_region_subproblem(
    grad: Tensor,
    hessian: Tensor,
    radius: float,
) -> TrustRegionSubproblemResult:
    """Solve the classical quadratic trust-region subproblem."""

    device = grad.device
    dtype = grad.dtype

    grad_np = grad.detach().cpu().numpy()
    hessian_np = hessian.detach().cpu().numpy()

    x0 = np.zeros_like(grad_np)

    def fun(x: np.ndarray) -> float:
        return 0.0

    def jac(x: np.ndarray) -> np.ndarray:
        return grad_np

    def hess(x: np.ndarray) -> np.ndarray:
        return hessian_np

    subproblem = IterativeSubproblem(
        x=x0,
        fun=fun,
        jac=jac,
        hess=hess,
        k_easy=_K_EASY,
        k_hard=_K_HARD,
        maxiter=_MAXITER
    )

    step_np, hits_boundary = subproblem.solve(radius)

    step = torch.as_tensor(
        step_np,
        dtype=dtype,
        device=device,
    )

    return TrustRegionSubproblemResult(
        step=step,
        hits_boundary=hits_boundary,
        multiplier=float(subproblem.lambda_current),
        iterations=subproblem.niter,
    )