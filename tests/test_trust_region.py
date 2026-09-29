import math
import tomllib
from pathlib import Path

import torch

from lago.trust_region import (
    TrustRegionConfig,
    TrustRegionState,
    cauchy_point,
    get_local_candidate,
    get_sr1_tr_config,
    improvement_ratio,
    is_box_feasible,
    predicted_improvement,
    project_onto_tr_ball,
    quadratic_model,
    sr1_update,
    update_radius,
)


def load_reference():
    path = Path("reference/data/trust_region_reference.toml")

    with path.open("rb") as f:
        return tomllib.load(f)

dtype = torch.float64

config = TrustRegionConfig(
    initial_radius=1.0,
    acceptance_threshold=0.1,
    shrink_threshold=0.25,
    expand_threshold=0.75,
    shrink_factor=0.5,
    expand_factor=2.0,
    max_radius=4.0,
)

## ---------------------------
## Quadratic model tests
## ---------------------------
def test_quadratic_model_matches_julia():
    data = load_reference()["quadratic_model"]

    f = torch.tensor(data["f_center"], dtype=dtype)
    grad = torch.tensor(data["gradient"], dtype=dtype)
    hessian = torch.tensor(data["hessian"], dtype=dtype)
    step = torch.tensor(data["step"], dtype=dtype)

    value = quadratic_model(f, grad, hessian, step)

    expected = torch.tensor(
        data["model_value"],
        dtype=dtype,
    )

    torch.testing.assert_close(value, expected)

## ---------------------------
## Predicted improvement tests
## ---------------------------
def test_predicted_improvement_matches_julia():
    data = load_reference()["quadratic_model"]

    f = torch.tensor(data["f_center"], dtype=dtype)
    grad = torch.tensor(data["gradient"], dtype=dtype)
    hessian = torch.tensor(data["hessian"], dtype=dtype)
    step = torch.tensor(data["step"], dtype=dtype)

    value = predicted_improvement(
        f,
        grad,
        hessian,
        step,
    )

    expected = torch.tensor(
        data["predicted_improvement"],
        dtype=dtype,
    )

    torch.testing.assert_close(value, expected)

## ---------------------------
## Predicted improvement tests
## ---------------------------
def test_improvement_ratio_matches_julia():
    reference = load_reference()

    model_data = reference["quadratic_model"]
    ratio_data = reference["improvement_ratio"]

    f_center = torch.tensor(
        model_data["f_center"],
        dtype=dtype,
    )

    f_trial = torch.tensor(
        ratio_data["f_trial"],
        dtype=dtype,
    )

    predicted_decrease = torch.tensor(
        model_data["predicted_improvement"],
        dtype=dtype,
    )

    rho = improvement_ratio(
        f_center,
        f_trial,
        predicted_decrease,
    )

    expected = torch.tensor(
        ratio_data["rho"],
        dtype=dtype,
    )

    torch.testing.assert_close(rho, expected)

## ----------------
## SR1 update tests
## ----------------
def test_sr1_update_matches_julia():
    reference = load_reference()

    model_data = reference["quadratic_model"]
    sr1_data = reference["accepted_sr1"]

    hessian = torch.tensor(
        model_data["hessian"],
        dtype=dtype,
    )

    step = torch.tensor(
        model_data["step"],
        dtype=dtype,
    )

    grad_center = torch.tensor(
        model_data["gradient"],
        dtype=dtype,
    )

    grad_trial = torch.tensor(
        sr1_data["gradient"],
        dtype=dtype,
    )

    grad_difference = grad_trial - grad_center

    updated_hessian = sr1_update(
        hessian=hessian,
        step=step,
        grad_difference=grad_difference,
    )

    expected = torch.tensor(
        sr1_data["hessian"],
        dtype=dtype,
    )

    torch.testing.assert_close(
        updated_hessian,
        expected,
    )

def test_sr1_skips_degenerate_update():
    hessian = torch.tensor(
        [[2.0, 0.3],
         [0.3, 1.0]],
        dtype=dtype,
    )

    step = torch.tensor(
        [0.2, -0.1],
        dtype=dtype,
    )

    grad_difference = hessian @ step

    updated_hessian = sr1_update(
        hessian,
        step,
        grad_difference,
    )

    torch.testing.assert_close(
        updated_hessian,
        hessian,
    )

## -------------------
## Radius update tests
## -------------------
def test_update_radius_shrinks():
    reference = load_reference()

    expected = reference["radius_update"]["shrink"]

    step = torch.tensor(
        [0.5, 0.0],
        dtype=dtype,
    )

    radius = update_radius(
        radius=1.0,
        rho=0.10,
        step=step,
        config=config
    )

    assert radius == expected

def test_update_radius_unchanged():
    reference = load_reference()

    expected = reference["radius_update"]["unchanged"]

    step = torch.tensor(
        [0.5, 0.0],
        dtype=dtype,
    )

    radius = update_radius(
        radius=1.0,
        rho=0.50,
        step=step,
        config=config
    )

    assert radius == expected

def test_update_radius_does_not_expand_for_interior_step():
    reference = load_reference()

    expected = reference["radius_update"]["good_interior"]

    step = torch.tensor(
        [0.5, 0.0],
        dtype=dtype,
    )

    radius = update_radius(
        radius=1.0,
        rho=0.90,
        step=step,
        config=config
    )

    assert radius == expected

def test_update_radius_expands_near_boundary():
    reference = load_reference()

    expected = reference["radius_update"]["expand"]

    step = torch.tensor(
        [0.9, 0.0],
        dtype=dtype,
    )

    radius = update_radius(
        radius=1.0,
        rho=0.90,
        step=step,
        config=config
    )

    assert radius == expected

def test_project_onto_tr_ball():
    step = torch.tensor(
        [3.0, 4.0],
        dtype=dtype,
    )

    projected = project_onto_tr_ball(
        step,
        radius=2.0,
    )

    torch.testing.assert_close(
        torch.linalg.norm(projected),
        torch.tensor(2.0, dtype=dtype),
    )

def test_local_candidate_uses_box_fallback():
    case = load_reference()[
        "box_constrained_subproblem"
    ]

    state = TrustRegionState(
        center=torch.tensor(
            case["center"],
            dtype=dtype,
        ),
        radius=case["radius"],
        f_center=torch.tensor(
            0.0,
            dtype=dtype,
        ),
        grad_center=torch.tensor(
            case["gradient"],
            dtype=dtype,
        ),
        hessian=torch.tensor(
            case["hessian"],
            dtype=dtype,
        ),
    )

    lower = torch.tensor(
        case["lower_bounds"],
        dtype=dtype,
    )

    upper = torch.tensor(
        case["upper_bounds"],
        dtype=dtype,
    )

    step = get_local_candidate(
        state=state,
        lower_bounds=lower,
        upper_bounds=upper,
    )

    expected = torch.tensor(
        case["get_tr_candidate_box_step"],
        dtype=dtype,
    )

    torch.testing.assert_close(
        step,
        expected,
        rtol=1e-5,
        atol=1e-6,
    )

    candidate = state.center + step

    assert is_box_feasible(
        candidate,
        lower,
        upper,
    )

    assert (
        torch.linalg.norm(step).item()
        <= state.radius * (1 + 1e-8)
    )

def test_local_candidate_uses_ball_solution_when_box_feasible():
    state = TrustRegionState(
        center=torch.tensor(
            [0.0, 0.0],
            dtype=dtype,
        ),
        radius=1.0,
        f_center=torch.tensor(
            0.0,
            dtype=dtype,
        ),
        grad_center=torch.tensor(
            [-0.2, 0.1],
            dtype=dtype,
        ),
        hessian=torch.tensor(
            [
                [2.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=dtype,
        ),
    )

    lower = torch.tensor(
        [-1.0, -1.0],
        dtype=dtype,
    )

    upper = torch.tensor(
        [1.0, 1.0],
        dtype=dtype,
    )

    step = get_local_candidate(
        state=state,
        lower_bounds=lower,
        upper_bounds=upper,
    )

    expected = torch.tensor(
        [0.1, -0.1],
        dtype=dtype,
    )

    torch.testing.assert_close(
        step,
        expected,
        rtol=1e-7,
        atol=1e-9,
    )

## ------------------
## Cauchy point tests
## ------------------
def test_cauchy_point_matches_julia():
    reference = load_reference()

    model_data = reference["quadratic_model"]
    cauchy_data = reference["cauchy_point"]

    grad = torch.tensor(
        model_data["gradient"],
        dtype=dtype,
    )

    hessian = torch.tensor(
        model_data["hessian"],
        dtype=dtype,
    )

    step = cauchy_point(
        grad=grad,
        hessian=hessian,
        radius=1.0,
    )

    expected = torch.tensor(
        cauchy_data["step"],
        dtype=dtype,
    )

    torch.testing.assert_close(step, expected)

    torch.testing.assert_close(
        torch.linalg.norm(step),
        torch.tensor(
            cauchy_data["step_norm"],
            dtype=dtype,
        ),
    )

def test_cauchy_point_zero_gradient():
    grad = torch.zeros(2, dtype=dtype)
    hessian = torch.eye(2, dtype=dtype)

    step = cauchy_point(
        grad=grad,
        hessian=hessian,
        radius=1.0,
    )

    torch.testing.assert_close(
        step,
        torch.zeros_like(grad),
    )

def test_cauchy_point_negative_curvature_hits_boundary():
    grad = torch.tensor(
        [1.0, 2.0],
        dtype=dtype,
    )

    hessian = -torch.eye(
        2,
        dtype=dtype,
    )

    radius = 0.7

    step = cauchy_point(
        grad=grad,
        hessian=hessian,
        radius=radius,
    )

    torch.testing.assert_close(
        torch.linalg.norm(step),
        torch.tensor(radius, dtype=dtype),
    )

def test_get_sr1_tr_config():
    bounds = torch.tensor(
        [[0.0, 0.0], [1.0, 1.0]],
        dtype=torch.float64,
    )

    config = get_sr1_tr_config(
        lengthscale=1.0,
        bounds=bounds,
    )

    diameter = math.sqrt(2.0)

    assert config.initial_radius == min(
        0.5,
        diameter / 8.0,
    )
    assert config.max_radius == diameter / 2.0