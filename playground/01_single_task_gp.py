import math
import torch
import gpytorch
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from gpytorch.kernels import MaternKernel, ScaleKernel

# use a GPU if available
device = torch.device("cpu")
dtype = torch.float64

# use regular spaced points on the interval [0, 1]
train_X = torch.linspace(0, 1, 15, dtype=dtype, device=device)
# training data needs to be explicitly multi-dimensional
train_X = train_X.unsqueeze(1) # -> (15, 1)

# sample observed values and add some synthetic noise
train_Y = torch.sin(train_X * (2 * math.pi))

# nugget
train_Yvar = torch.full_like(train_Y, 1e-9)

covar_module = ScaleKernel(
    MaternKernel(
        nu=2.5,
        ard_num_dims=train_X.shape[-1],
    )
)

with gpytorch.settings.min_fixed_noise(double_value=1e-12):
    model = SingleTaskGP(train_X=train_X, train_Y=train_Y, train_Yvar = train_Yvar, covar_module = covar_module)
# model.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(1e-5))

mll = ExactMarginalLogLikelihood(likelihood=model.likelihood, model=model)

# set mll and all submodules to the specified dtype and device
mll = mll.to(train_X)

fit_gpytorch_mll(mll)
model.eval()

with torch.no_grad():
    posterior_train = model.posterior(train_X)

print(
    torch.cat(
        [
            train_Y,
            posterior_train.mean,
            posterior_train.variance,
        ],
        dim=1,
    )
)

print("Y std:")
print(model.outcome_transform.stdvs)

print("Y variance:")
print(model.outcome_transform.stdvs.square())

print("requested noise, standardized:")
print(
    train_Yvar[0]
    / model.outcome_transform.stdvs.square().squeeze()
)

print("noise actually stored by likelihood:")
print(model.likelihood.noise[0])
