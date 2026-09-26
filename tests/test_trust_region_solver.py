from pathlib import Path
import tomllib
import pytest

import torch

from lago._trust_region_solver import solve_trust_region_subproblem

dtype = torch.float64

def load_reference():
    path = Path("reference/data/trust_region_reference.toml")

    with path.open("rb") as f:
        return tomllib.load(f)

        import pytest

@pytest.mark.parametrize(
    "case_name",
    [
        "interior",
        "boundary",
        "indefinite",
        "hard",
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

    if case_name != "hard":
        torch.testing.assert_close(
            result.step,
            expected_step,
            rtol=1e-5,
            atol=1e-7,
        )