# Given a fitted GP, compute \nabla \mu(x), and \nabla^2 \mu(x), and check
# numerically with FD.
import math
import torch
import gpytorch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.func import grad

device = torch.device("cpu")
dtype = torch.float64

def gradient(f,x):
    return grad(f)(x)

def hessian(f, x):
    return torch.autograd.functional.hessian(
        f,
        x,
        vectorize=False,
    ) #this is 1.5 x faster than the double jacrev with chunk_size = 1.

x = torch.linspace(0, 1, 4, dtype=dtype, device=device)
x1, x2 = torch.meshgrid(x, x, indexing="ij")
print(x1.shape)
print(x2.shape)

# However, BoTorch does not want two 4x4 arrays, it wants 16 x 2 dimensions.
train_X = torch.stack(
    [x1.flatten(), x2.flatten()],
    dim=-1,
) # (16,2)

def f(X):
    x1 = X[..., 0] # ... takes all dims before, here equiv to [:, 0] as only 2D.
    x2 = X[..., 1]

    return (
        torch.sin(2 * math.pi * x1)
        + 0.5 * torch.cos(2 * math.pi * x2)
    )

train_Y = f(train_X)
print(train_Y.shape)
# this is (16,), but we want R^{n x 1}
train_Y = train_Y.unsqueeze(-1)
print(train_Y.shape)

print("Training sizes")
print("X:", train_X.shape, "Y:",train_Y.shape)

# GP construction

train_Yvar = torch.full_like(train_Y, 1e-9) # nugget

covar_module = ScaleKernel( # kernel
    MaternKernel(
        nu = 2.5,
        ard_num_dims=train_X.shape[-1], # this is now 2
        )
)

with gpytorch.settings.min_fixed_noise(double_value=1e-12):
    model = SingleTaskGP(
        train_X = train_X,
        train_Y = train_Y,
        train_Yvar = train_Yvar,
        covar_module = covar_module
    )

mll = ExactMarginalLogLikelihood( # creating the likelihood model to optimize
    model.likelihood,
    model,
)

fit_gpytorch_mll(mll)
model.eval()
print("lengthscales:")
print(model.covar_module.base_kernel.lengthscale)

print("\noutputscale:")
print(model.covar_module.outputscale)

# trying at a given point
x0 = torch.tensor(
    [0.37, 0.61],
    dtype=dtype,
    device=device,
)
x0_batch = x0.unsqueeze(0)
posterior = model.posterior(x0_batch)
print("posterior mean:")
print(posterior.mean)

print("posterior variance:")
print(posterior.variance)

def posterior_mean(x):
    x_batch = x.unsqueeze(0)
    posterior = model.posterior(x_batch)
    return posterior.mean.squeeze()

mu0 = posterior_mean(x0)
print(mu0)
print(mu0.shape)

# Now onto gradients and hessians:
gp_grad = gradient(posterior_mean, x0)
gp_hessian = hessian(posterior_mean, x0)

print("posterior mean:")
print(posterior_mean(x0))

print("\nposterior gradient:")
print(gp_grad)

print("\nposterior Hessian:")
print(gp_hessian)

symmetry_error = torch.linalg.norm(
    gp_hessian - gp_hessian.T
)

print("\nHessian symmetry error:")
print(symmetry_error)
