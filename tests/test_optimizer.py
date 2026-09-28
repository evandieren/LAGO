import math

import torch

import lago.optimizer as optimizer
from lago.gp import build_value_gp
from lago.iteration import IterationProposal, IterationResult
from lago.optimizer import LAGOState
from lago.trust_region import TrustRegionConfig, TrustRegionState


dtype = torch.float64


def objective(x: torch.Tensor) -> torch.Tensor:
    return x @ x


def gradient(x: torch.Tensor) -> torch.Tensor:
    return 2.0 * x


def make_config() -> TrustRegionConfig:
    return TrustRegionConfig(
        initial_radius=0.25,
        acceptance_threshold=5e-4,
        shrink_threshold=0.25,
        expand_threshold=0.75,
        shrink_factor=0.5,
        expand_factor=2.0,
        max_radius=1.0,
    )


def make_initial_data() -> tuple[torch.Tensor, torch.Tensor]:
    X = torch.tensor(
        [
            [0.8, 0.8],
            [0.6, 0.9],
            [0.9, 0.6],
        ],
        dtype=dtype,
    )

    Y = torch.stack(
        [objective(x) for x in X]
    ).unsqueeze(-1)

    return X, Y


def make_dummy_lago_state() -> LAGOState:
    X = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 1.0],
        ],
        dtype=dtype,
    )
    Y = torch.tensor(
        [
            [1.0],
            [2.0],
        ],
        dtype=dtype,
    )

    model = build_value_gp(X, Y)

    trust_region = TrustRegionState(
        center=X[0].clone(),
        radius=0.25,
        f_center=Y[0, 0].clone(),
        grad_center=torch.tensor(
            [1.0, 0.0],
            dtype=dtype,
        ),
        hessian=torch.eye(
            2,
            dtype=dtype,
        ),
    )

    return LAGOState(
        model=model,
        trust_region=trust_region,
        tr_config=make_config(),
        active_X=X.clone(),
        active_Y=Y.clone(),
        archive=optimizer.make_initial_archive(X, Y),
    )


def test_initialize_lago_builds_archive_gp_and_tr(
    monkeypatch,
):
    initial_X, initial_Y = make_initial_data()
    config = make_config()

    x_informed = torch.tensor(
        [0.1, 0.2],
        dtype=dtype,
    )

    fit_calls = 0

    def fake_fit_value_gp(model):
        nonlocal fit_calls
        fit_calls += 1
        return model

    config = make_config()

    monkeypatch.setattr(
        optimizer,
        "get_sr1_tr_config",
        lambda lengthscale, bounds: config,
    )

    monkeypatch.setattr(
        optimizer,
        "fit_value_gp",
        fake_fit_value_gp,
    )

    monkeypatch.setattr(
        optimizer,
        "get_informed_candidate",
        lambda *args, **kwargs: x_informed.clone(),
    )

    state = optimizer.initialize_lago(
        initial_X=initial_X,
        initial_Y=initial_Y,
        objective=objective,
        gradient=gradient,
        bounds=torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 1.0],
            ],
            dtype=dtype,
        ),
    )

    n0 = initial_X.shape[0]

    # Hyperparameters are fitted once on D0.
    assert fit_calls == 1

    # x^I is evaluated and added to both the active GP data and archive.
    assert state.active_X.shape == (n0 + 1, 2)
    assert state.active_Y.shape == (n0 + 1, 1)
    assert state.archive.X.shape == (n0 + 1, 2)
    assert state.archive.Y.shape == (n0 + 1, 1)

    torch.testing.assert_close(
        state.active_X[-1],
        x_informed,
    )
    torch.testing.assert_close(
        state.active_Y[-1, 0],
        objective(x_informed),
    )

    # The informed point is the best observed point, so the TR starts there.
    torch.testing.assert_close(
        state.trust_region.center,
        x_informed,
    )
    torch.testing.assert_close(
        state.trust_region.f_center,
        objective(x_informed),
    )
    torch.testing.assert_close(
        state.trust_region.grad_center,
        gradient(x_informed),
    )

    assert state.trust_region.radius == config.initial_radius
    assert not state.trust_region.terminated

    # Only the TR center has a gradient evaluation at initialization.
    assert state.archive.has_gradient.sum().item() == 1
    assert state.archive.has_gradient[-1].item()

    torch.testing.assert_close(
        state.archive.gradients[-1],
        gradient(x_informed),
    )

    # The conditioned GP contains D1.
    torch.testing.assert_close(
        state.model.train_inputs[0],
        state.active_X,
    )

    assert state.trust_region.hessian.shape == (2, 2)
    assert torch.all(
        torch.isfinite(state.trust_region.hessian)
    )

    assert state.iteration == 0
    assert state.low_ei_count == 0


def test_run_lago_stops_after_completed_iterations_on_low_ei(
    monkeypatch,
):
    epsilon_t = 1e-12
    initial_state = make_dummy_lago_state()

    initial_archive_size = initial_state.archive.X.shape[0]
    initial_active_size = initial_state.active_X.shape[0]

    monkeypatch.setattr(
        optimizer,
        "initialize_lago",
        lambda **kwargs: initial_state,
    )

    log_eis = iter(
        [
            math.log(1e-13),
            math.log(1e-13),
            math.log(1e-2),   # reset the consecutive-low-EI counter
            math.log(1e-13),
            math.log(1e-13),
            math.log(1e-13),
            math.log(1e-13),
            math.log(1e-13),  # fifth consecutive low EI -> stop
        ]
    )

    proposal_calls = 0

    def fake_propose_iteration(**kwargs):
        nonlocal proposal_calls
        proposal_calls += 1

        return IterationProposal(
            state=kwargs["state"],
            step=torch.zeros(
                2,
                dtype=dtype,
            ),
            local_improvement=torch.tensor(
                1e-13,
                dtype=dtype,
            ),
            x_global=torch.tensor(
                [0.5, 0.5],
                dtype=dtype,
            ),
            log_ei=torch.tensor(
                next(log_eis),
                dtype=dtype,
            ),
        )

    monkeypatch.setattr(
        optimizer,
        "propose_iteration",
        fake_propose_iteration,
    )

    execute_calls = 0

    def fake_execute_iteration(**kwargs):
        nonlocal execute_calls
        execute_calls += 1

        proposal = kwargs["proposal"]

        return IterationResult(
            state=proposal.state,
            model=kwargs["model"],
            active_X=kwargs["active_X"],
            active_Y=kwargs["active_Y"],
            point=proposal.x_global,
            value=torch.tensor(
                0.25,
                dtype=dtype,
            ),
            gradient=None,
            branch="global",
            accepted=None,
        )

    monkeypatch.setattr(
        optimizer,
        "execute_iteration",
        fake_execute_iteration,
    )

    initial_X, initial_Y = make_initial_data()

    state = optimizer.run_lago(
        initial_X=initial_X,
        initial_Y=initial_Y,
        objective=objective,
        gradient=gradient,
        bounds=torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 1.0],
            ],
            dtype=dtype,
        ),
        max_iterations=20,
        epsilon_t=epsilon_t,
        n_low_ei=5,
        refit_interval=100,
    )

    # Eight proposals were made and all eight iterations were completed.
    # The fifth consecutive low-EI result causes the NEXT iteration
    # to be rejected at the loop boundary.
    assert proposal_calls == 8
    assert execute_calls == 8

    assert state.low_ei_count == 5

    # Initial state has iteration=0, so eight completed loop
    # iterations leave us at iteration 8.
    assert state.iteration == 8

    assert (
        state.archive.X.shape[0]
        == initial_archive_size + execute_calls
    )

    # The fake execution leaves active GP data unchanged, showing that the
    # archive and active GP dataset are independent.
    assert state.active_X.shape[0] == initial_active_size
    assert state.archive.X.shape[0] > state.active_X.shape[0]


def test_run_lago_completes_iteration_before_budget_stop(
    monkeypatch,
):
    initial_state = make_dummy_lago_state()

    monkeypatch.setattr(
        optimizer,
        "initialize_lago",
        lambda **kwargs: initial_state,
    )

    proposal_calls = 0

    def fake_propose_iteration(**kwargs):
        nonlocal proposal_calls
        proposal_calls += 1

        return IterationProposal(
            state=kwargs["state"],
            step=torch.tensor(
                [-0.1, 0.0],
                dtype=dtype,
            ),
            local_improvement=torch.tensor(
                1.0,
                dtype=dtype,
            ),
            x_global=torch.tensor(
                [0.5, 0.5],
                dtype=dtype,
            ),
            log_ei=torch.tensor(
                0.0,
                dtype=dtype,
            ),
        )

    monkeypatch.setattr(
        optimizer,
        "propose_iteration",
        fake_propose_iteration,
    )

    execute_calls = 0

    def fake_execute_iteration(**kwargs):
        nonlocal execute_calls
        execute_calls += 1

        proposal = kwargs["proposal"]

        return IterationResult(
            state=proposal.state,
            model=kwargs["model"],
            active_X=kwargs["active_X"],
            active_Y=kwargs["active_Y"],
            point=torch.tensor(
                [0.4, 0.4],
                dtype=dtype,
            ),
            value=torch.tensor(
                0.5,
                dtype=dtype,
            ),
            gradient=torch.tensor(
                [0.8, 0.8],
                dtype=dtype,
            ),
            branch="local",
            accepted=True,
        )

    monkeypatch.setattr(
        optimizer,
        "execute_iteration",
        fake_execute_iteration,
    )

    initial_cost = initial_state.archive.cost(
        gradient_cost=2,
    )

    state = optimizer.run_lago(
        initial_X=torch.empty((0, 2), dtype=dtype),
        initial_Y=torch.empty((0, 1), dtype=dtype),
        objective=objective,
        gradient=gradient,
        bounds=torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 1.0],
            ],
            dtype=dtype,
        ),
        evaluation_budget=initial_cost + 1,
        gradient_cost=2,
        max_iterations=20,
        refit_interval=100,
    )

    # We entered the iteration because cost was still below budget.
    assert proposal_calls == 1
    assert execute_calls == 1

    # One local evaluation costs:
    #     1 function + 2 gradient-equivalent evaluations = 3.
    assert state.archive.cost(2) == initial_cost + 3

    # Therefore overshooting the nominal budget is intentional.
    assert state.archive.cost(2) > initial_cost + 1