using LAGO_BO
using AbstractBayesOpt
using LinearAlgebra
using Optim
using TOML


# ------------------------------------------------------------------
# Fixed trust-region configuration
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


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

# TOML prefers nested vectors over Julia matrices.
matrix_to_rows(A) = [collect(row) for row in eachrow(A)]


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


# ------------------------------------------------------------------
# Basic trust-region state
# ------------------------------------------------------------------

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


# ------------------------------------------------------------------
# Quadratic model
# ------------------------------------------------------------------

s = [0.2, -0.1]

tr = make_tr()

q = LAGO_BO.quad_surrogate(s, tr)

predicted_improvement =
    tr.f_center - q


# ------------------------------------------------------------------
# Cauchy point
# ------------------------------------------------------------------

cauchy = LAGO_BO.cauchy_point(
    g,
    H,
    1.0,
)

cauchy_norm = norm(cauchy)


# ------------------------------------------------------------------
# Improvement ratio
# ------------------------------------------------------------------

f_trial = 0.8

ir = LAGO_BO.compute_ir(
    tr,
    s,
    f_trial,
)


# ------------------------------------------------------------------
# Radius updates
# ------------------------------------------------------------------

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


# ------------------------------------------------------------------
# Accepted SR1 update
# ------------------------------------------------------------------

x_trial = center + s

y_trial = [
    0.8,   # f(x_trial)
   -0.5,   # df/dx1
    1.7,   # df/dx2
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


# ------------------------------------------------------------------
# Rejected SR1 update
# ------------------------------------------------------------------

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


# ------------------------------------------------------------------
# Classical ball-only trust-region subproblem
# ------------------------------------------------------------------

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


# ------------------------------------------------------------------
# Box-constrained trust-region fallback
# ------------------------------------------------------------------
#
# The ball-only solution moves +0.5 in x1:
#
#     center + step = [0.9, 0.5] + [0.5, 0.0]
#                   = [1.4, 0.5]
#
# which violates the upper box bound x1 <= 1.
#
# The constrained solution should therefore saturate the box at
# approximately step = [0.1, 0.0].
# ------------------------------------------------------------------

box_center = [0.9, 0.5]

box_lower = [0.0, 0.0]
box_upper = [1.0, 1.0]

box_gradient = [-1.0, 0.0]

box_hessian = [
    1.0  0.0
    0.0  1.0
]

box_radius = 0.5

box_domain = ContinuousDomain(
    box_lower,
    box_upper,
)

box_tr = TrustRegion(
    copy(box_center),
    box_radius,
    0.0,
    copy(box_gradient),
    copy(box_hessian),
    config,
    false,
    0.0,
    copy(box_gradient),
    copy(box_hessian),
)


# First solve the ordinary ball-only TR problem.
ball_step, ball_reached_solution =
    LAGO_BO.get_tr_candidate(
        box_tr;
        verbose=false,
    )

ball_candidate =
    box_center .+ ball_step

ball_feasible =
    all(box_lower .<= ball_candidate) &&
    all(ball_candidate .<= box_upper)


# Now explicitly call the constrained fallback.
constrained_step =
    LAGO_BO.get_tr_candidate_constrained(
        box_tr,
        box_domain;
        verbose=false,
    )

constrained_candidate =
    box_center .+ constrained_step

constrained_model_value =
    LAGO_BO.quad_surrogate(
        constrained_step,
        box_tr,
    )


# Finally call the complete wrapper. Since the ball-only solution
# violates the box, this should return the constrained solution.
box_step =
    LAGO_BO.get_tr_candidate_box(
        box_tr,
        box_domain;
        verbose=false,
    )


# ------------------------------------------------------------------
# Sanity checks for the box-constrained fixture
# ------------------------------------------------------------------

@assert ball_reached_solution

@assert !ball_feasible

@assert norm(ball_step) <= box_radius * (1 + 1e-8)

@assert all(box_lower .<= constrained_candidate)
@assert all(constrained_candidate .<= box_upper)

@assert norm(constrained_step) <= box_radius * (1 + 1e-8)

@assert isapprox(
    box_step,
    constrained_step;
    atol=1e-8,
    rtol=1e-8,
)


# ------------------------------------------------------------------
# Reference data
# ------------------------------------------------------------------

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

    "box_constrained_subproblem" => Dict(
        "center" => box_center,
        "lower_bounds" => box_lower,
        "upper_bounds" => box_upper,
        "gradient" => box_gradient,
        "hessian" => matrix_to_rows(box_hessian),
        "radius" => box_radius,

        "ball_step" => ball_step,
        "ball_candidate" => ball_candidate,
        "ball_feasible" => ball_feasible,
        "ball_reached_solution" => ball_reached_solution,

        "constrained_step" => constrained_step,
        "constrained_candidate" => constrained_candidate,
        "constrained_model_value" => constrained_model_value,

        "get_tr_candidate_box_step" => box_step,
    ),
)


# ------------------------------------------------------------------
# Write TOML
# ------------------------------------------------------------------

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

println()
println("Box-constrained fixture:")
println("  ball step             = ", ball_step)
println("  ball candidate        = ", ball_candidate)
println("  ball feasible         = ", ball_feasible)
println("  constrained step      = ", constrained_step)
println("  constrained candidate = ", constrained_candidate)
println("  constrained model     = ", constrained_model_value)