from pathlib import Path
import tomllib

import numpy as np
from scipy.optimize._trustregion_exact import IterativeSubproblem


# ---------------------------------------------------------------------
# Load Julia reference data
# ---------------------------------------------------------------------

reference_path = Path("reference/data/trust_region_reference.toml")

with reference_path.open("rb") as f:
    reference = tomllib.load(f)

cases = reference["trust_region_subproblem"]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def quadratic_value(
    grad: np.ndarray,
    hessian: np.ndarray,
    step: np.ndarray,
) -> float:
    """Evaluate q(s) = g^T s + 0.5 s^T H s."""
    return float(
        grad @ step
        + 0.5 * step @ hessian @ step
    )


def solve_with_scipy(
    grad: np.ndarray,
    hessian: np.ndarray,
    radius: float,
    k_hard: float,
):
    """Solve the classical trust-region subproblem with SciPy."""

    x0 = np.zeros_like(grad)

    # IterativeSubproblem is normally called by scipy.optimize.minimize.
    # Here g and H are already fixed, so these functions are constant.
    def fun(x):
        return 0.0

    def jac(x):
        return grad

    def hess(x):
        return hessian

    subproblem = IterativeSubproblem(
        x=x0,
        fun=fun,
        jac=jac,
        hess=hess,
        k_easy=1e-10,
        k_hard=k_hard,
        maxiter=1000,
    )

    step, hits_boundary = subproblem.solve(radius)

    return step, hits_boundary, subproblem


# ---------------------------------------------------------------------
# Hard-case tolerance sweep
# ---------------------------------------------------------------------

k_hard_values = [
    2e-1,
    1e-2,
    1e-4,
    1e-6,
    1e-8,
    1e-10,
]

for name in ["indefinite", "hard"]:
    case = cases[name]

    grad = np.asarray(
        case["gradient"],
        dtype=float,
    )

    hessian = np.asarray(
        case["hessian"],
        dtype=float,
    )

    radius = float(case["radius"])

    print()
    print("=" * 90)
    print(name.upper())
    print("=" * 90)

    print(f"gradient:\n{grad}")
    print(f"hessian:\n{hessian}")
    print(f"radius: {radius}")
    print()

    print(
        f"{'k_hard':>12}"
        f"{'niter':>10}"
        f"{'||s||':>16}"
        f"{'lambda':>20}"
        f"{'q(s)':>20}"
        f"{'boundary':>12}"
    )

    print("-" * 90)

    for k_hard in k_hard_values:
        step, hits_boundary, subproblem = solve_with_scipy(
            grad=grad,
            hessian=hessian,
            radius=radius,
            k_hard=k_hard,
        )

        step_norm = np.linalg.norm(step)
        model_value = quadratic_value(
            grad,
            hessian,
            step,
        )

        print(
            f"{k_hard:12.1e}"
            f"{subproblem.niter:10d}"
            f"{step_norm:16.12f}"
            f"{subproblem.lambda_current:20.12f}"
            f"{model_value:20.12f}"
            f"{str(hits_boundary):>12}"
        )

    print()

    # Print the most accurate run in detail.
    step, hits_boundary, subproblem = solve_with_scipy(
        grad=grad,
        hessian=hessian,
        radius=radius,
        k_hard=k_hard_values[-1],
    )

    print(f"Step for k_hard = {k_hard_values[-1]:.1e}:")
    print(step)

    print(f"norm:   {np.linalg.norm(step)}")
    print(f"lambda: {subproblem.lambda_current}")
    print(
        "q(s):   "
        f"{quadratic_value(grad, hessian, step)}"
    )
    print(f"niter:  {subproblem.niter}")
    print(f"hits boundary: {hits_boundary}")