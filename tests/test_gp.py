from functools import partial

import torch
from torch import Tensor

from lago.autodiff import hessian
from lago.gp import (
    build_value_gp,
    fit_value_gp,
    posterior_mean,
    posterior_mean_hessian,
    posterior_variance,
    update_value_gp,
)


dtype = torch.float64


def make_training_data() -> tuple[Tensor, Tensor]:
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


def finite_difference_hessian(
    f,
    x: Tensor,
    h: float = 1e-4,
) -> Tensor:
    n = x.numel()

    H = torch.empty(
        (n, n),
        dtype=x.dtype,
        device=x.device,
    )

    f0 = f(x)

    for i in range(n):
        e_i = torch.zeros_like(x)
        e_i[i] = h

        H[i, i] = (
            f(x + e_i)
            - 2.0 * f0
            + f(x - e_i)
        ) / h**2

        for j in range(i + 1, n):
            e_j = torch.zeros_like(x)
            e_j[j] = h

            H_ij = (
                f(x + e_i + e_j)
                - f(x + e_i - e_j)
                - f(x - e_i + e_j)
                + f(x - e_i - e_j)
            ) / (4.0 * h**2)

            H[i, j] = H_ij
            H[j, i] = H_ij

    return H


def test_value_gp_shapes():
    train_X, train_Y = make_training_data()

    model = build_value_gp(
        train_X,
        train_Y,
    )

    assert model.train_inputs[0].shape == (4, 2)
    assert model.train_targets.shape == (4,)


def test_value_gp_uses_initial_mean():
    train_X, train_Y = make_training_data()

    model = build_value_gp(
        train_X,
        train_Y,
    )

    expected = train_Y.mean()

    torch.testing.assert_close(
        model.mean_module.constant.squeeze(),
        expected,
    )

    assert not model.mean_module.constant.requires_grad


def test_value_gp_initial_hyperparameters():
    train_X, train_Y = make_training_data()

    model = build_value_gp(
        train_X,
        train_Y,
    )

    torch.testing.assert_close(
        model.covar_module.base_kernel.lengthscale.squeeze(),
        torch.tensor(1.0, dtype=dtype),
    )

    torch.testing.assert_close(
        model.covar_module.outputscale.squeeze(),
        torch.tensor(1.0, dtype=dtype),
    )


def test_value_gp_uses_fixed_nugget():
    train_X, train_Y = make_training_data()

    model = build_value_gp(
        train_X,
        train_Y,
        nugget=1e-9,
    )

    expected = torch.full(
        (train_X.shape[0],),
        1e-9,
        dtype=dtype,
    )

    torch.testing.assert_close(
        model.likelihood.noise,
        expected,
    )


def test_value_gp_trainable_parameters():
    train_X, train_Y = make_training_data()

    model = build_value_gp(
        train_X,
        train_Y,
    )

    trainable = {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }

    assert trainable == {
        "covar_module.raw_outputscale",
        "covar_module.base_kernel.raw_lengthscale",
    }


def test_fit_value_gp():
    train_X, train_Y = make_training_data()

    model = build_value_gp(
        train_X,
        train_Y,
    )

    prior_mean_before = (
        model.mean_module
        .constant
        .detach()
        .clone()
    )

    fit_value_gp(model)

    lengthscale = (
        model.covar_module
        .base_kernel
        .lengthscale
        .detach()
        .squeeze()
    )

    outputscale = (
        model.covar_module
        .outputscale
        .detach()
        .squeeze()
    )

    assert torch.isfinite(lengthscale)
    assert torch.isfinite(outputscale)

    assert lengthscale > 0
    assert outputscale > 0

    torch.testing.assert_close(
        model.mean_module.constant,
        prior_mean_before,
    )


def test_posterior_shapes_for_batch():
    train_X, train_Y = make_training_data()

    model = build_value_gp(
        train_X,
        train_Y,
    )

    X = torch.tensor(
        [
            [0.1, 0.2],
            [0.7, 0.6],
            [0.9, 0.3],
        ],
        dtype=dtype,
    )

    mean = posterior_mean(model, X)
    variance = posterior_variance(model, X)

    assert mean.shape == (3, 1)
    assert variance.shape == (3, 1)

    assert torch.all(torch.isfinite(mean))
    assert torch.all(torch.isfinite(variance))
    assert torch.all(variance >= 0)


def test_posterior_shapes_for_single_point():
    train_X, train_Y = make_training_data()

    model = build_value_gp(
        train_X,
        train_Y,
    )

    x = torch.tensor(
        [0.3, 0.4],
        dtype=dtype,
    )

    mean = posterior_mean(model, x)
    variance = posterior_variance(model, x)

    assert mean.ndim == 0
    assert variance.ndim == 0

    assert torch.isfinite(mean)
    assert torch.isfinite(variance)
    assert variance >= 0


def test_posterior_mean_nearly_interpolates_training_data():
    train_X, train_Y = make_training_data()

    model = build_value_gp(
        train_X,
        train_Y,
    )

    mean = posterior_mean(
        model,
        train_X,
    )

    torch.testing.assert_close(
        mean,
        train_Y,
        rtol=1e-5,
        atol=1e-6,
    )


def test_posterior_mean_hessian_at_training_point():
    train_X, train_Y = make_training_data()

    model = build_value_gp(
        train_X,
        train_Y,
    )

    # Exact coincidence with a GP training point.
    x = train_X[1].clone()

    mean = partial(
        posterior_mean,
        model,
    )

    H = posterior_mean_hessian(
        model,
        x,
    )

    H_fd = finite_difference_hessian(
        mean,
        x,
        h=1e-4,
    )

    assert H.shape == (2, 2)
    assert torch.all(torch.isfinite(H))

    torch.testing.assert_close(
        H,
        H.T,
        rtol=1e-10,
        atol=1e-12,
    )

    torch.testing.assert_close(
        H,
        H_fd,
        rtol=1e-3,
        atol=1e-4,
    )


def test_posterior_mean_hessian_matches_autograd_away_from_data():
    train_X, train_Y = make_training_data()

    model = build_value_gp(
        train_X,
        train_Y,
    )

    x = torch.tensor(
        [0.3, 0.4],
        dtype=dtype,
    )

    mean = partial(
        posterior_mean,
        model,
    )

    H = posterior_mean_hessian(
        model,
        x,
    )

    H_ad = hessian(
        mean,
        x,
    )

    torch.testing.assert_close(
        H,
        H_ad,
        rtol=1e-6,
        atol=1e-8,
    )

def test_ard_posterior_mean_hessian_at_training_point():
    train_X, train_Y = make_training_data()

    model = build_value_gp(
        train_X,
        train_Y,
        use_ard=True,
    )

    model.covar_module.base_kernel.initialize(
        lengthscale=torch.tensor(
            [0.7, 1.4],
            dtype=dtype,
        )
    )

    x = train_X[1].clone()

    mean = partial(
        posterior_mean,
        model,
    )

    H = posterior_mean_hessian(
        model,
        x,
    )

    H_fd = finite_difference_hessian(
        mean,
        x,
        h=1e-4,
    )

    torch.testing.assert_close(
        H,
        H_fd,
        rtol=1e-3,
        atol=1e-4,
    )

def test_ard_posterior_mean_hessian_matches_autograd_away_from_data():
    train_X, train_Y = make_training_data()

    model = build_value_gp(
        train_X,
        train_Y,
        use_ard=True,
    )

    model.covar_module.base_kernel.initialize(
        lengthscale=torch.tensor(
            [0.7, 1.4],
            dtype=dtype,
        )
    )

    x = torch.tensor(
        [0.3, 0.4],
        dtype=dtype,
    )

    mean = partial(
        posterior_mean,
        model,
    )

    H = posterior_mean_hessian(
        model,
        x,
    )

    H_ad = hessian(
        mean,
        x,
    )

    torch.testing.assert_close(
        H,
        H_ad,
        rtol=1e-6,
        atol=1e-8,
    )

def test_update_value_gp_replaces_training_data_and_preserves_mean():
    train_X, train_Y = make_training_data()

    model = build_value_gp(
        train_X,
        train_Y,
    )

    initial_mean = (
        model.mean_module
        .constant
        .detach()
        .clone()
    )

    new_X = torch.tensor(
        [
            [0.0, 0.0],
            [0.2, 0.8],
            [0.4, 0.6],
        ],
        dtype=dtype,
    )

    new_Y = torch.tensor(
        [
            [10.0],
            [20.0],
            [30.0],
        ],
        dtype=dtype,
    )

    updated = update_value_gp(
        model,
        new_X,
        new_Y,
    )

    torch.testing.assert_close(
        updated.train_inputs[0],
        new_X,
    )

    torch.testing.assert_close(
        updated.train_targets,
        new_Y.squeeze(-1),
    )

    torch.testing.assert_close(
        updated.mean_module.constant,
        initial_mean,
    )

    # Make sure the test would catch recomputing
    # the prior mean from the new active data.
    assert not torch.isclose(
        updated.mean_module.constant.squeeze(),
        new_Y.mean(),
    )

def test_update_value_gp_preserves_hyperparameters():
    train_X, train_Y = make_training_data()

    model = build_value_gp(
        train_X,
        train_Y,
        use_ard=True,
    )

    model.covar_module.base_kernel.initialize(
        lengthscale=torch.tensor(
            [0.7, 1.4],
            dtype=dtype,
        )
    )

    model.covar_module.initialize(
        outputscale=2.3,
    )

    new_X = train_X[:3]
    new_Y = train_Y[:3]

    updated = update_value_gp(
        model,
        new_X,
        new_Y,
    )

    torch.testing.assert_close(
        updated.covar_module.base_kernel.lengthscale,
        model.covar_module.base_kernel.lengthscale,
    )

    torch.testing.assert_close(
        updated.covar_module.outputscale,
        model.covar_module.outputscale,
    )

    assert (
        updated.covar_module
        .base_kernel
        .ard_num_dims
        == train_X.shape[-1]
    )

def test_update_value_gp_preserves_nugget():
    train_X, train_Y = make_training_data()

    model = build_value_gp(
        train_X,
        train_Y,
        nugget=1e-9,
    )

    new_X = train_X[:2]
    new_Y = train_Y[:2]

    updated = update_value_gp(
        model,
        new_X,
        new_Y,
    )

    expected = torch.full(
        (2,),
        1e-9,
        dtype=dtype,
    )

    torch.testing.assert_close(
        updated.likelihood.noise,
        expected,
    )

def test_update_value_gp_can_refit():
    train_X, train_Y = make_training_data()

    model = build_value_gp(
        train_X,
        train_Y,
    )

    initial_mean = (
        model.mean_module
        .constant
        .detach()
        .clone()
    )

    new_X = train_X[:3]
    new_Y = train_Y[:3]

    updated = update_value_gp(
        model,
        new_X,
        new_Y,
        refit=True,
    )

    lengthscale = (
        updated.covar_module
        .base_kernel
        .lengthscale
        .detach()
    )

    outputscale = (
        updated.covar_module
        .outputscale
        .detach()
    )

    assert torch.all(torch.isfinite(lengthscale))
    assert torch.all(lengthscale > 0)

    assert torch.isfinite(outputscale)
    assert outputscale > 0

    torch.testing.assert_close(
        updated.mean_module.constant,
        initial_mean,
    )