from dataclasses import dataclass

import numpy as np
import torch
from scipy.optimize import Bounds, NonlinearConstraint, minimize
from scipy.optimize._trustregion_exact import IterativeSubproblem
from torch import Tensor


_K_EASY = 1e-10
_K_HARD = 1e-6
_MAXITER = 100

# COnstants for generic constrained fallback:
_BOX_GTOL = 1e-12
_BOX_XTOL = 1e-12
_BOX_BARRIER_TOL = 1e-12
_BOX_INITIAL_BARRIER = 1e-8
_BOX_MAXITER = 1000

@dataclass(frozen=True)
class TrustRegionSubproblemResult:
    step: Tensor
    hits_boundary: bool
    multiplier: float
    iterations: int

@dataclass(frozen=True)
class BoxTrustRegionSubproblemResult:
    step: Tensor
    success: bool
    iterations: int
    objective_value: float
    constraint_violation: float

def solve_trust_region_subproblem(
    grad: Tensor,
    hessian: Tensor,
    radius: float,
) -> TrustRegionSubproblemResult:
    """
    Solve the classical quadratic trust-region subproblem.
    
    The problem is formulated as:
        min_s  g^T s + 0.5 s^T H s
        s.t.   ||s||_2 <= radius

    Arguments:
        grad: Gradient vector g.
        hessian: Hessian matrix H.
        radius: Trust-region radius.


    Returns:
        TrustRegionSubproblemResult: Result of the optimization, including the step,
        whether the boundary was hit, the Lagrange multiplier, and the number of iterations.
    """

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

def solve_box_trust_region_subproblem(
    grad: Tensor,
    hessian: Tensor,
    radius: float,
    step_lower: Tensor,
    step_upper: Tensor,
) -> BoxTrustRegionSubproblemResult:
    """
    Solve the quadratic TR model subject to ball and box constraints.

    The problem is formulated as:
    
        min_s  g^T s + 0.5 s^T H s
        s.t.   ||s||_2 <= radius
               step_lower <= s <= step_upper

    Arguments:
        grad: Gradient vector g.
        hessian: Hessian matrix H.
        radius: Trust-region radius.
        step_lower: Lower bounds for the step.
        step_upper: Upper bounds for the step.
    
    Returns:
        BoxTrustRegionSubproblemResult: Result of the optimization, including the step,
        success flag, number of iterations, objective value, and constraint violation.
    """

    device = grad.device
    dtype = grad.dtype

    grad_np = grad.detach().cpu().numpy()
    hessian_np = hessian.detach().cpu().numpy()

    lower_np = step_lower.detach().cpu().numpy()
    upper_np = step_upper.detach().cpu().numpy()

    n = grad_np.size

    eps = 1e-8
    initial_guess = np.clip(
        np.zeros_like(grad_np),
        lower_np + eps,
        upper_np - eps,
    )

    def objective(step: np.ndarray) -> float:
        return float(
            grad_np @ step
            + 0.5 * step @ hessian_np @ step
        )

    def objective_grad(step: np.ndarray) -> np.ndarray:
        return grad_np + hessian_np @ step

    def objective_hess(step: np.ndarray) -> np.ndarray:
        return hessian_np

    # c(s) = ||s||² <= radius²
    def tr_constraint(step: np.ndarray) -> np.ndarray:
        return np.array([step @ step])

    def tr_constraint_jac(step: np.ndarray) -> np.ndarray:
        return (2.0 * step)[None, :]

    def tr_constraint_hess(
        step: np.ndarray,
        multiplier: np.ndarray,
    ) -> np.ndarray:
        return 2.0 * multiplier[0] * np.eye(n)

    nonlinear_constraint = NonlinearConstraint(
        fun=tr_constraint,
        lb=-np.inf,
        ub=radius**2,
        jac=tr_constraint_jac,
        hess=tr_constraint_hess,
        keep_feasible=True,
    )

    bounds = Bounds(
        lower_np,
        upper_np,
        keep_feasible=True,
    )

    result = minimize(
        fun=objective,
        x0=initial_guess,
        method="trust-constr",
        jac=objective_grad,
        hess=objective_hess,
        bounds=bounds,
        constraints=[nonlinear_constraint],
        options={
            "gtol": _BOX_GTOL,
            "xtol": _BOX_XTOL,
            "barrier_tol": _BOX_BARRIER_TOL,
            "initial_barrier_parameter": _BOX_INITIAL_BARRIER,
            "initial_barrier_tolerance": _BOX_INITIAL_BARRIER,
            "maxiter": _BOX_MAXITER,
            "verbose": 0,
        },
    )

    step = torch.as_tensor(
        result.x,
        dtype=dtype,
        device=device,
    )

    return BoxTrustRegionSubproblemResult(
        step=step,
        success=bool(result.success),
        iterations=int(result.nit),
        objective_value=float(result.fun),
        constraint_violation=float(result.constr_violation),
    )

