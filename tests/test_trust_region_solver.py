import tomllib
from pathlib import Path

import pytest
import torch

from lago._trust_region_solver import (
    solve_box_trust_region_subproblem,
    solve_trust_region_subproblem,
)

dtype = torch.float64

def load_reference():
    path = Path("reference/data/trust_region_reference.toml")

    with path.open("rb") as f:
        return tomllib.load(f)


@pytest.mark.parametrize(
    "case_name",
    [
        "interior",
        "boundary",
        "indefinite",
    ],
)
def test_trust_region_solver_matches_reference(case_name):
    case = load_reference()[
        "trust_region_subproblem"
    ][case_name]

    grad = torch.tensor(
        case["gradient"],
        dtype=dtype,
    )

    hessian = torch.tensor(
        case["hessian"],
        dtype=dtype,
    )

    result = solve_trust_region_subproblem(
        grad=grad,
        hessian=hessian,
        radius=case["radius"],
    )

    expected_step = torch.tensor(
        case["step"],
        dtype=dtype,
    )

    torch.testing.assert_close(
        result.step,
        expected_step,
        rtol=1e-5,
        atol=1e-7,
    )

    expected_value = (
        grad @ expected_step
        + 0.5 * expected_step @ hessian @ expected_step
    )

    actual_value = (
        grad @ result.step
        + 0.5 * result.step @ hessian @ result.step
    )

    torch.testing.assert_close(
        actual_value,
        expected_value,
        rtol=1e-6,
        atol=1e-8,
    )

    assert (
        torch.linalg.norm(result.step).item()
        <= case["radius"] * (1 + 1e-8)
    )

def test_box_trust_region_solver_matches_julia():
    case = load_reference()[
        "box_constrained_subproblem"
    ]

    center = torch.tensor(
        case["center"],
        dtype=dtype,
    )

    lower = torch.tensor(
        case["lower_bounds"],
        dtype=dtype,
    )

    upper = torch.tensor(
        case["upper_bounds"],
        dtype=dtype,
    )

    grad = torch.tensor(
        case["gradient"],
        dtype=dtype,
    )

    hessian = torch.tensor(
        case["hessian"],
        dtype=dtype,
    )

    radius = case["radius"]

    result = solve_box_trust_region_subproblem(
        grad=grad,
        hessian=hessian,
        radius=radius,
        step_lower=lower - center,
        step_upper=upper - center,
    )

    expected = torch.tensor(
        case["constrained_step"],
        dtype=dtype,
    )

    torch.testing.assert_close(
        result.step,
        expected,
        rtol=1e-5,
        atol=1e-6,
    )

    assert torch.linalg.norm(result.step) <= radius * (
        1 + 1e-8
    )

    candidate = center + result.step

    assert torch.all(candidate >= lower - 1e-8)
    assert torch.all(candidate <= upper + 1e-8)

def test_trust_region_solver_hard_case():
    grad = torch.tensor(
        [0.0, 1.0],
        dtype=dtype,
    )

    hessian = torch.tensor(
        [
            [-1.0, 0.0],
            [0.0, 2.0],
        ],
        dtype=dtype,
    )

    radius = 1.0

    result = solve_trust_region_subproblem(
        grad=grad,
        hessian=hessian,
        radius=radius,
    )

    # The hard-case minimizers are
    #
    #     s = [±sqrt(8/9), -1/3]
    #
    # with optimal model value -2/3.

    expected_value = torch.tensor(
        -2.0 / 3.0,
        dtype=dtype,
    )

    actual_value = (
        grad @ result.step
        + 0.5
        * result.step
        @ hessian
        @ result.step
    )

    torch.testing.assert_close(
        actual_value,
        expected_value,
        rtol=1e-6,
        atol=1e-8,
    )

    torch.testing.assert_close(
        torch.linalg.norm(result.step),
        torch.tensor(radius, dtype=dtype),
        rtol=1e-6,
        atol=1e-8,
    )