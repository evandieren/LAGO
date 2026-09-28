import pytest
import torch

from botorch.acquisition import LogExpectedImprovement

from lago.global_search import get_global_candidate
from lago.gp import build_value_gp


dtype = torch.float64


def make_training_data():
    train_X = torch.tensor(
        [
            [0.0, 0.0],
            [0.5, 0.2],
            [0.2, 0.8],
            [0.8, 0.7],
        ],
        dtype=dtype,
    )

    train_Y = torch.tensor(
        [
            [1.0],
            [0.4],
            [0.7],
            [0.2],
        ],
        dtype=dtype,
    )

    return train_X, train_Y


def make_bounds():
    return torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 1.0],
        ],
        dtype=dtype,
    )


def test_global_candidate_is_feasible():
    torch.manual_seed(0)

    train_X, train_Y = make_training_data()

    model = build_value_gp(
        train_X,
        train_Y,
    )

    bounds = make_bounds()

    center = torch.tensor(
        [0.5, 0.5],
        dtype=dtype,
    )

    radius = 0.25

    candidate, log_ei_value = get_global_candidate(
        model=model,
        bounds=bounds,
        best_f=train_Y.min(),
        center=center,
        radius=radius,
        num_restarts=10,
        raw_samples=256,
    )

    assert candidate.shape == (2,)
    assert log_ei_value.ndim == 0

    assert torch.all(torch.isfinite(candidate))
    assert torch.isfinite(log_ei_value)

    assert torch.all(
        candidate >= bounds[0]
    )

    assert torch.all(
        candidate <= bounds[1]
    )

    distance = torch.linalg.norm(
        candidate - center
    )

    assert distance >= radius - 1e-8


def test_global_candidate_returns_correct_log_ei_value():
    torch.manual_seed(0)

    train_X, train_Y = make_training_data()

    model = build_value_gp(
        train_X,
        train_Y,
    )

    bounds = make_bounds()

    center = torch.tensor(
        [0.5, 0.5],
        dtype=dtype,
    )

    radius = 0.25

    candidate, actual_log_ei = get_global_candidate(
        model=model,
        bounds=bounds,
        best_f=train_Y.min(),
        center=center,
        radius=radius,
        num_restarts=10,
        raw_samples=256,
    )

    log_ei = LogExpectedImprovement(
        model=model,
        best_f=train_Y.min(),
        maximize=False,
    )

    expected_log_ei = log_ei(
        candidate.unsqueeze(-2)
    ).squeeze()

    torch.testing.assert_close(
        actual_log_ei,
        expected_log_ei,
        rtol=1e-8,
        atol=1e-10,
    )

def test_global_candidate_raises_if_no_feasible_points():
    torch.manual_seed(0)

    train_X, train_Y = make_training_data()

    model = build_value_gp(
        train_X,
        train_Y,
    )

    bounds = make_bounds()

    center = torch.tensor(
        [0.5, 0.5],
        dtype=dtype,
    )

    # Entire [0,1]^2 box lies inside this trust region.
    radius = 1.0

    with pytest.raises(
        RuntimeError,
        match="Not enough feasible initial points",
    ):
        get_global_candidate(
            model=model,
            bounds=bounds,
            best_f=train_Y.min(),
            center=center,
            radius=radius,
            num_restarts=10,
            raw_samples=256,
        )

def test_global_candidate_beats_feasible_grid_baseline():
    torch.manual_seed(0)

    train_X, train_Y = make_training_data()
    model = build_value_gp(train_X, train_Y)

    bounds = make_bounds()

    center = torch.tensor(
        [0.5, 0.5],
        dtype=dtype,
    )
    radius = 0.25

    _, candidate_log_ei = get_global_candidate(
        model=model,
        bounds=bounds,
        best_f=train_Y.min(),
        center=center,
        radius=radius,
        num_restarts=10,
        raw_samples=256,
    )

    grid_1d = torch.linspace(
        0.0,
        1.0,
        51,
        dtype=dtype,
    )

    X1, X2 = torch.meshgrid(
        grid_1d,
        grid_1d,
        indexing="ij",
    )

    grid = torch.stack(
        [X1.reshape(-1), X2.reshape(-1)],
        dim=-1,
    )

    feasible = (
        torch.linalg.norm(
            grid - center,
            dim=-1,
        )
        >= radius
    )

    feasible_grid = grid[feasible]

    log_ei = LogExpectedImprovement(
        model=model,
        best_f=train_Y.min(),
        maximize=False,
    )

    with torch.no_grad():
        grid_values = log_ei(
            feasible_grid.unsqueeze(-2)
        )

    best_grid_value = grid_values.max()

    assert candidate_log_ei >= best_grid_value - 1e-5

def test_global_candidate_beats_feasible_grid_baseline():
    torch.manual_seed(0)

    train_X, train_Y = make_training_data()
    model = build_value_gp(train_X, train_Y)

    bounds = make_bounds()

    log_ei = LogExpectedImprovement(
        model=model,
        best_f=train_Y.min(),
        maximize=False,
    )

    # Build a dense grid over the box.
    grid_1d = torch.linspace(
        0.0,
        1.0,
        51,
        dtype=dtype,
    )

    X1, X2 = torch.meshgrid(
        grid_1d,
        grid_1d,
        indexing="ij",
    )

    grid = torch.stack(
        [
            X1.reshape(-1),
            X2.reshape(-1),
        ],
        dim=-1,
    )

    with torch.no_grad():
        grid_values = log_ei(
            grid.unsqueeze(-2)
        )

    # Put the TR around the unconstrained grid maximizer.
    center = grid[grid_values.argmax()].clone()
    radius = 0.2

    feasible = (
        torch.linalg.norm(
            grid - center,
            dim=-1,
        )
        >= radius
    )

    feasible_grid_values = grid_values[feasible]

    best_feasible_grid_value = (
        feasible_grid_values.max()
    )

    candidate, candidate_log_ei = get_global_candidate(
        model=model,
        bounds=bounds,
        best_f=train_Y.min(),
        center=center,
        radius=radius,
        num_restarts=10,
        raw_samples=256,
    )

    # The TR constraint is active: its center was the
    # unconstrained grid optimum.
    distance = torch.linalg.norm(
        candidate - center
    )

    assert distance >= radius - 1e-8

    # Continuous constrained optimization should be at least
    # as good as the coarse feasible grid baseline.
    assert (
        candidate_log_ei
        >= best_feasible_grid_value - 1e-5
    )