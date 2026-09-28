import gpytorch
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch import Tensor
from torch.func import grad


_DEFAULT_NUGGET = 1e-9


def build_value_gp(
    train_X: Tensor,
    train_Y: Tensor,
    *,
    prior_mean: Tensor | None = None,
    nugget: float = _DEFAULT_NUGGET,
    use_ard: bool = False,
) -> SingleTaskGP:
    """
    Build the value-only GP used by LAGO-BO.

    The model uses:
    - an isotropic Matern-5/2 kernel,
    - a fixed constant prior mean,
    - a fixed numerical nugget,
    - no input or outcome transform.

    Hyperparameters are not fitted by this function.
    """

    if train_X.ndim != 2:
        raise ValueError(
            "train_X must have shape (n, d)."
        )

    if train_Y.ndim != 2 or train_Y.shape[-1] != 1:
        raise ValueError(
            "train_Y must have shape (n, 1)."
        )

    if train_X.shape[0] != train_Y.shape[0]:
        raise ValueError(
            "train_X and train_Y must contain the same "
            "number of observations."
        )

    if train_X.dtype != torch.float64:
        raise ValueError(
            "LAGO GP models currently require torch.float64."
        )

    if prior_mean is None:
        prior_mean = train_Y.mean()

    mean_module = ConstantMean()
    mean_module.initialize(
        constant=prior_mean.item(),
    )

    # The prior mean is prescribed by LAGO, not estimated by MLE.
    mean_module.constant.requires_grad_(False)
    if use_ard:
        base_kernel = MaternKernel(
            nu=2.5,
            ard_num_dims=train_X.shape[-1],
        )
    else:
        base_kernel = MaternKernel(
            nu=2.5,
        )

    base_kernel.initialize(
        lengthscale=1.0,
    )
    covar_module = ScaleKernel(
        base_kernel,
    )
    covar_module.initialize(
        outputscale=1.0,
    )

    train_Yvar = torch.full_like(
        train_Y,
        nugget,
    )

    with gpytorch.settings.min_fixed_noise(double_value=1e-12):
        model = SingleTaskGP(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            mean_module=mean_module,
            covar_module=covar_module,
            outcome_transform=None,
            input_transform=None,
        )

    return model


def fit_value_gp(
    model: SingleTaskGP,
) -> SingleTaskGP:
    """Fit the GP kernel hyperparameters by marginal likelihood."""

    mll = ExactMarginalLogLikelihood(
        model.likelihood,
        model,
    )

    fit_gpytorch_mll(mll)

    return model

def posterior_mean(
    model: SingleTaskGP,
    X: Tensor,
) -> Tensor:
    """
    Posterior mean at one point or a batch of points.

    posterior_mean(model, x)   # x: (d,)   -> scalar
    posterior_mean(model, X)   # X: (n,d)  -> (n,1)
    """

    single_point = X.ndim == 1

    if single_point:
        X = X.unsqueeze(0)

    mean = model.posterior(X).mean

    if single_point:
        return mean.squeeze()

    return mean

def posterior_variance(
    model: SingleTaskGP,
    X: Tensor,
) -> Tensor:
    """Posterior variance at one point or a batch of points."""

    single_point = X.ndim == 1

    if single_point:
        X = X.unsqueeze(0)

    variance = model.posterior(X).variance

    if single_point:
        return variance.squeeze()

    return variance

def posterior_mean_hessian(
    model: SingleTaskGP,
    x: Tensor,
) -> Tensor:
    """Hessian of the Matérn-5/2 GP posterior mean at x."""

    if x.ndim != 1:
        raise ValueError("x must have shape (d,).")

    train_X = model.train_inputs[0]
    train_Y = model.train_targets

    lengthscale = (
        model.covar_module
        .base_kernel
        .lengthscale
        .squeeze()
    )

    outputscale = (
        model.covar_module
        .outputscale
        .squeeze()
    )

    K = model.covar_module(
        train_X,
        train_X,
    ).to_dense()

    noise = torch.diag(
        model.likelihood.noise
    )

    mean_train = model.mean_module(
        train_X
    )

    alpha = torch.linalg.solve(
        K + noise,
        train_Y - mean_train,
    )

    delta = x.unsqueeze(0) - train_X
    r = torch.linalg.vector_norm(
        delta,
        dim=-1,
    )

    a = torch.sqrt(
        torch.tensor(
            5.0,
            dtype=x.dtype,
            device=x.device,
        )
    ) / lengthscale

    exp_term = torch.exp(-a * r)

    identity = torch.eye(
        x.numel(),
        dtype=x.dtype,
        device=x.device,
    )

    outer = (
        delta.unsqueeze(-1)
        * delta.unsqueeze(-2)
    )

    H_kernel = (
        outputscale
        * exp_term[:, None, None]
        * (
            a**4 / 3.0 * outer
            - (
                a**2 / 3.0
                * (1.0 + a * r)
            )[:, None, None]
            * identity
        )
    )

    return torch.einsum(
        "n,nij->ij",
        alpha,
        H_kernel,
    )