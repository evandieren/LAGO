import torch
from torch.quasirandom import SobolEngine
from botorch.test_functions import Branin

from lago.autodiff import gradient
from lago.optimizer import run_lago
from lago.trust_region import TrustRegionConfig

import matplotlib.pyplot as plt

dtype = torch.float64

problem = Branin().to(dtype=dtype)

bounds = problem.bounds

def objective(x):
    return problem(x)

def objective_gradient(x):
    return gradient(objective, x)


sobol = SobolEngine(
    dimension=2,
    scramble=True,
    seed=0,
)

unit_X = sobol.draw(10).to(dtype=dtype)

initial_X = (
    bounds[0]
    + (bounds[1] - bounds[0]) * unit_X
)

initial_Y = problem(initial_X).unsqueeze(-1)

d = bounds.shape[-1]

state = run_lago(
    initial_X=initial_X,
    initial_Y=initial_Y,
    objective=objective,
    gradient=objective_gradient,
    bounds=bounds,
    evaluation_budget= 210*d,
    gradient_cost=d,
)

best_index = state.archive.Y.squeeze(-1).argmin()

print("iterations:", state.iteration)
print("archive size:", len(state.archive.X))
print("active GP size:", len(state.active_X))

print("best x:", state.archive.X[best_index])
print("best f:", state.archive.Y[best_index].item())

print("TR center:", state.trust_region.center)
print("TR f:", state.trust_region.f_center.item())
print("TR radius:", state.trust_region.radius)

print("initial radius:", state.tr_config.initial_radius)
print("max radius:", state.tr_config.max_radius)
print("iterations:", state.iteration)
print("f evals:", state.archive.n_function_evals)
print("g evals:", state.archive.n_gradient_evals)
print("cost:", state.archive.cost(d))

best_values = torch.cummin(state.archive.Y.squeeze(-1), dim=0).values

error = torch.clamp(best_values - problem.optimal_value, min=torch.finfo(dtype).eps)
n_evals = torch.arange(
    1,
    len(best_values) + 1,
)

fig, ax = plt.subplots()

ax.semilogy(
    n_evals.cpu(),
    error.cpu(),
)

ax.set_xlabel("Function evaluations")
ax.set_ylabel(r"$f_{\mathrm{best}} - f^\star$")

plt.show()