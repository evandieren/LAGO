import matplotlib.pyplot as plt
import torch
from botorch.test_functions import Hartmann
from botorch.utils.sampling import draw_sobol_samples

from lago.autodiff import gradient
from lago.optimizer import run_lago
from lago.plotting import equivalent_evaluation_costs

dtype = torch.float64

problem = Hartmann(dim=6).to(dtype=dtype)
bounds = problem.bounds
d = bounds.shape[1]


def objective(x):
    return problem(x)


def objective_gradient(x):
    return gradient(objective, x)

f_star = problem.optimal_value
print("true minimum:", f_star)

# Paper setup: 5d initial points.
n_initial = 5 * d

initial_X = draw_sobol_samples(
    bounds=bounds,
    n=n_initial,
    q=1,
).squeeze(1)

initial_Y = problem(initial_X).unsqueeze(-1)


state = run_lago(
    initial_X=initial_X,
    initial_Y=initial_Y,
    objective=objective,
    gradient=objective_gradient,
    bounds=bounds,
    gradient_cost=d,
    evaluation_budget= 210*d,
    num_restarts=100,
    raw_samples=10000
)


best_index = state.archive.Y.argmin()

print("iterations:", state.iteration)
print("archive size:", state.archive.X.shape[0])
print("active GP size:", state.active_X.shape[0])
print("best x:", state.archive.X[best_index])
print("best f:", state.archive.Y[best_index].item())
print("true minimum:", problem.optimal_value)
print("TR center:", state.trust_region.center)
print("TR f:", state.trust_region.f_center.item())
print("TR radius:", state.trust_region.radius)
print("initial radius:", state.tr_config.initial_radius)
print("max radius:", state.tr_config.max_radius)
print("f evals:", state.archive.n_function_evals)
print("g evals:", state.archive.n_gradient_evals)
print("cost:", state.archive.cost(d))


cost = equivalent_evaluation_costs(
    state,
    gradient_cost=d,
)

best_values = torch.cummin(
    state.archive.Y.squeeze(-1),
    dim=0,
).values

error = torch.clamp(
    best_values - f_star,
    min=torch.finfo(best_values.dtype).eps,
)

fig, ax = plt.subplots()

ax.semilogy(
    cost.cpu(),
    error.cpu(),
)

ax.set_xlabel("Equivalent function evaluations")
ax.set_ylabel(r"$f_{\mathrm{best}} - f^\star$")
ax.set_title("LAGO-BO — Hartmann 6D")

plt.show()