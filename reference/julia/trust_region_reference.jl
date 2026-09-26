using LAGO_BO
using LinearAlgebra
using Optim
using TOML


# ------------------------------------------------------------------
# Fixed trust-region state
# ------------------------------------------------------------------

config = TRConfig(
    1.0,   # r0
    0.1,   # η   : acceptance threshold
    0.25,  # η1  : shrink threshold
    0.75,  # η2  : expansion threshold
    0.5,   # γ1  : shrink factor
    2.0,   # γ2  : expansion factor
    4.0,   # r_max
)

center = [0.0, 0.0]

f_center = 1.0

g = [-1.0, 2.0]

H = [
    2.0  0.3
    0.3  1.0
]


function make_tr()
    return TrustRegion(
        copy(center),
        1.0,
        f_center,
        copy(g),
        copy(H),
        config,
        false,
        f_center,
        copy(g),
        copy(H),
    )
end


# Helper because TOML prefers nested vectors over Julia matrices.
matrix_to_rows(A) = [collect(row) for row in eachrow(A)]

s = [0.2, -0.1]

tr = make_tr()

q = LAGO_BO.quad_surrogate(s, tr)
predicted_improvement = tr.f_center - q

cauchy = LAGO_BO.cauchy_point(
    g,
    H,
    1.0,
)
cauchy_norm = norm(cauchy)

f_trial = 0.8

ir = LAGO_BO.compute_ir(
    tr,
    s,
    f_trial,
)

radius_shrink = LAGO_BO.update_radius(
    1.0,
    0.10,
    [0.5, 0.0],
    [0.0, 0.0],
    config.η₁,
    config.η₂,
    config.γ₁,
    config.γ₂,
    config.r_max,
)

radius_unchanged = LAGO_BO.update_radius(
    1.0,
    0.50,
    [0.5, 0.0],
    [0.0, 0.0],
    config.η₁,
    config.η₂,
    config.γ₁,
    config.γ₂,
    config.r_max,
)

radius_good_interior = LAGO_BO.update_radius(
    1.0,
    0.90,
    [0.5, 0.0],
    [0.0, 0.0],
    config.η₁,
    config.η₂,
    config.γ₁,
    config.γ₂,
    config.r_max,
)

radius_expand = LAGO_BO.update_radius(
    1.0,
    0.90,
    [0.9, 0.0],
    [0.0, 0.0],
    config.η₁,
    config.η₂,
    config.γ₁,
    config.γ₂,
    config.r_max,
)

x_trial = center + s

y_trial = [
    0.8,   # f(x_trial)
   -0.5,   # ∂f/∂x1
    1.7,   # ∂f/∂x2
]

mu = [0.0]
sigma = [1.0]

tr_accepted = make_tr()

tr_accepted, accepted = LAGO_BO.update_TR(
    tr_accepted,
    ir,
    x_trial,
    y_trial,
    mu,
    sigma,
)

println("accepted = ", accepted)
println("center = ", tr_accepted.center)
println("radius = ", tr_accepted.radius)
println("H = ")
display(tr_accepted.Hk)


tr_rejected = make_tr()

f_trial_bad = 1.1

ir_bad = LAGO_BO.compute_ir(
    tr_rejected,
    s,
    f_trial_bad,
)

y_trial_bad = [
    f_trial_bad,
   -0.5,
    1.7,
]

tr_rejected, accepted_bad = LAGO_BO.update_TR(
    tr_rejected,
    ir_bad,
    x_trial,
    y_trial_bad,
    mu,
    sigma,
)


# TR subproblem solutions
function solve_reference(g, H, radius)
    s = zeros(length(g))

    _, interior, lambda, _, reached_solution =
        Optim.solve_tr_subproblem!(
            g,
            H,
            radius,
            s,
        )

    return Dict(
        "gradient" => g,
        "hessian" => matrix_to_rows(H),
        "radius" => radius,
        "step" => copy(s),
        "step_norm" => norm(s),
        "lambda" => lambda,
        "interior" => interior,
        "reached_solution" => reached_solution,
    )
end

tr_interior = solve_reference(
    [-0.2, 0.1],
    [
        2.0  0.0
        0.0  1.0
    ],
    1.0,
)

tr_boundary = solve_reference(
    [-1.0, 2.0],
    [
        2.0  0.3
        0.3  1.0
    ],
    1.0,
)

tr_indefinite = solve_reference(
    [0.2, 1.0],
    [
        -1.0  0.0
         0.0  2.0
    ],
    1.0,
)

tr_hard = solve_reference(
    [0.0, 1.0],
    [
        -1.0  0.0
         0.0  2.0
    ],
    1.0,
)

data = Dict(
    "quadratic_model" => Dict(
        "center" => center,
        "f_center" => f_center,
        "gradient" => g,
        "hessian" => matrix_to_rows(H),
        "step" => s,
        "model_value" => q,
        "predicted_improvement" => predicted_improvement,
    ),

    "cauchy_point" => Dict(
        "step" => cauchy,
        "step_norm" => cauchy_norm,
    ),

    "improvement_ratio" => Dict(
        "f_trial" => f_trial,
        "rho" => ir,
    ),

    "radius_update" => Dict(
        "shrink" => radius_shrink,
        "unchanged" => radius_unchanged,
        "good_interior" => radius_good_interior,
        "expand" => radius_expand,
    ),

    "accepted_sr1" => Dict(
        "accepted" => accepted,
        "center" => tr_accepted.center,
        "radius" => tr_accepted.radius,
        "gradient" => tr_accepted.∇f_center,
        "hessian" => matrix_to_rows(tr_accepted.Hk),
    ),

    "rejected_sr1" => Dict(
        "rho" => ir_bad,
        "accepted" => accepted_bad,
        "center" => tr_rejected.center,
        "radius" => tr_rejected.radius,
        "gradient" => tr_rejected.∇f_center,
        "hessian" => matrix_to_rows(tr_rejected.Hk),
    ),

    "trust_region_subproblem" => Dict(
        "interior" => tr_interior,
        "boundary" => tr_boundary,
        "indefinite" => tr_indefinite,
        "hard" => tr_hard,
    ),
)

output_path = normpath(
    joinpath(
        @__DIR__,
        "..",
        "data",
        "trust_region_reference.toml",
    )
)

mkpath(dirname(output_path))

open(output_path, "w") do io
    TOML.print(io, data)
end

println("Wrote reference data to:")
println(output_path)
