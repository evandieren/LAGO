import torch

from lago.iteration import (
    perform_local_step,
)
from lago.trust_region import (
    TrustRegionConfig,
    TrustRegionState,
)

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


def objective(x):
    return x @ x


def gradient(x):
    return 2.0 * x

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


def test_accepted_local_step_updates_state_and_filters_gp_data():
    state = TrustRegionState(
        center=torch.tensor(
            [1.0, 0.0],
            dtype=dtype,
        ),
        radius=1.0,
        f_center=torch.tensor(
            1.0,
            dtype=dtype,
        ),
        grad_center=torch.tensor(
            [2.0, 0.0],
            dtype=dtype,
        ),
        hessian=2.0 * torch.eye(
            2,
            dtype=dtype,
        ),
    )

    step = torch.tensor(
        [-0.5, 0.0],
        dtype=dtype,
    )

    active_X = torch.tensor(
        [
            [0.0, 0.0],
            [0.7, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
        ],
        dtype=dtype,
    )

    active_Y = torch.tensor(
        [
            [0.0],
            [0.49],
            [1.0],
            [4.0],
        ],
        dtype=dtype,
    )

    result = perform_local_step(
        state=state,
        step=step,
        objective=objective,
        gradient=gradient,
        active_X=active_X,
        active_Y=active_Y,
        config=config,
        lengthscale=0.5,
        nu=0.5,
    )

    assert result.accepted

    torch.testing.assert_close(
        result.state.center,
        torch.tensor(
            [0.5, 0.0],
            dtype=dtype,
        ),
    )

    torch.testing.assert_close(
        result.state.f_center,
        torch.tensor(
            0.25,
            dtype=dtype,
        ),
    )

    torch.testing.assert_close(
        result.state.grad_center,
        torch.tensor(
            [1.0, 0.0],
            dtype=dtype,
        ),
    )

    # The nearby point at x=0.7 is removed.
    assert not torch.any(
        torch.all(
            result.active_X
            == torch.tensor(
                [0.7, 0.0],
                dtype=dtype,
            ),
            dim=-1,
        )
    )

    # New center is assimilated.
    assert torch.any(
        torch.all(
            result.active_X
            == result.state.center,
            dim=-1,
        )
    )


def test_rejected_local_step_keeps_center_but_updates_sr1():
    state = TrustRegionState(
        center=torch.tensor(
            [1.0, 0.0],
            dtype=dtype,
        ),
        radius=1.0,
        f_center=torch.tensor(
            1.0,
            dtype=dtype,
        ),
        grad_center=torch.tensor(
            [2.0, 0.0],
            dtype=dtype,
        ),
        hessian=torch.tensor(
            [
                [-10.0, 0.0],
                [0.0, 2.0],
            ],
            dtype=dtype,
        ),
    )

    step = torch.tensor(
        [0.5, 0.0],
        dtype=dtype,
    )

    active_X = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=dtype,
    )

    active_Y = torch.tensor(
        [
            [0.0],
            [1.0],
        ],
        dtype=dtype,
    )

    result = perform_local_step(
        state=state,
        step=step,
        objective=objective,
        gradient=gradient,
        active_X=active_X,
        active_Y=active_Y,
        config=config,
        lengthscale=0.5,
        nu=0.5,
    )

    assert not result.accepted

    torch.testing.assert_close(
        result.state.center,
        state.center,
    )

    assert result.state.radius == 0.5

    # Despite rejection, the trial gradient updates SR1.
    torch.testing.assert_close(
        result.state.hessian,
        2.0 * torch.eye(
            2,
            dtype=dtype,
        ),
    )

    # Rejected local points are not assimilated into the GP.
    torch.testing.assert_close(
        result.active_X,
        active_X,
    )

    torch.testing.assert_close(
        result.active_Y,
        active_Y,
    )