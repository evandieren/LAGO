from pathlib import Path
import tomllib

import pytest
import torch

from lago.gp import (
    build_value_gp,
    posterior_mean,
    posterior_mean_hessian,
    posterior_variance,
)


dtype = torch.float64


def load_reference():
    path = Path(
        "reference/data/gp_reference.toml"
    )

    with path.open("rb") as f:
        return tomllib.load(f)


@pytest.mark.parametrize(
    "case_name, ard",
    [
        ("isotropic", False),
        ("ard", True),
    ],
)
def test_gp_matches_julia_reference(
    case_name,
    ard,
):
    reference = load_reference()

    training = reference["training_data"]
    hyperparameters = reference["hyperparameters"]
    case = reference[case_name]

    train_X = torch.tensor(
        training["X"],
        dtype=dtype,
    )

    train_Y = torch.tensor(
        training["Y"],
        dtype=dtype,
    ).unsqueeze(-1)

    query_points = torch.tensor(
        reference["query_points"],
        dtype=dtype,
    )

    prior_mean = torch.tensor(
        hyperparameters["prior_mean"],
        dtype=dtype,
    )

    model = build_value_gp(
        train_X,
        train_Y,
        prior_mean=prior_mean,
        nugget=hyperparameters["nugget"],
        use_ard=ard,
    )

    lengthscale = torch.tensor(
        case["lengthscale"],
        dtype=dtype,
    )

    if not ard:
        lengthscale = lengthscale[0]

    model.covar_module.base_kernel.initialize(
        lengthscale=lengthscale,
    )

    model.covar_module.initialize(
        outputscale=hyperparameters["outputscale"],
    )

    expected_mean = torch.tensor(
        case["posterior_mean"],
        dtype=dtype,
    )

    expected_variance = torch.tensor(
        case["posterior_variance"],
        dtype=dtype,
    )

    expected_hessian = torch.tensor(
        case["posterior_hessian"],
        dtype=dtype,
    )

    actual_mean = posterior_mean(model, query_points).squeeze(-1)

    actual_variance = posterior_variance(model, query_points).squeeze(-1)

    actual_hessian = torch.stack(
        [
            posterior_mean_hessian(model, x)
            for x in query_points
        ]
    )

    torch.testing.assert_close(
        actual_mean,
        expected_mean,
        rtol=1e-7,
        atol=1e-7,
    )

    torch.testing.assert_close(
        actual_variance,
        expected_variance,
        rtol=1e-7,
        atol=1e-7,
    )

    torch.testing.assert_close(
        actual_hessian,
        expected_hessian,
        rtol=1e-7,
        atol=1e-7,
    )