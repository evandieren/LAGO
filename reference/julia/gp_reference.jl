using LinearAlgebra
using TOML
using Statistics

# ---------------------------------------------------------------------------
# Matérn-5/2 kernel
# ---------------------------------------------------------------------------

function matern52_kernel(
    x,
    y;
    lengthscale,
    outputscale,
)
    δ = x - y
    r = norm(δ ./ lengthscale)

    return outputscale *
           (1 + sqrt(5) * r + 5r^2 / 3) *
           exp(-sqrt(5) * r)
end


function matern52_hessian(
    x,
    y;
    lengthscale,
    outputscale,
)
    δ = x - y

    inv_ell2 = 1.0 ./ lengthscale.^2

    scaled_δ = δ ./ lengthscale
    r = norm(scaled_δ)

    qδ = δ .* inv_ell2
    Q = Diagonal(inv_ell2)

    return outputscale *
           exp(-sqrt(5) * r) *
           (
               (25 / 3) .* (qδ * qδ') -
               (5 / 3) * (1 + sqrt(5) * r) .* Q
           )
end


# ---------------------------------------------------------------------------
# GP posterior
# ---------------------------------------------------------------------------

function build_gp_reference(
    X,
    y,
    query_points;
    prior_mean,
    lengthscale,
    outputscale,
    nugget,
)
    n = length(X)

    K = Matrix{Float64}(undef, n, n)

    for i in 1:n
        for j in i:n
            value = matern52_kernel(
                X[i],
                X[j];
                lengthscale=lengthscale,
                outputscale=outputscale,
            )

            K[i, j] = value
            K[j, i] = value
        end
    end

    Ky = K + nugget * I

    α = Ky \ (y .- prior_mean)

    means = Float64[]
    variances = Float64[]
    hessians = Matrix{Float64}[]

    for x in query_points
        kx = [
            matern52_kernel(
                x,
                xi;
                lengthscale=lengthscale,
                outputscale=outputscale,
            )
            for xi in X
        ]

        mean_x =
            prior_mean + dot(kx, α)

        variance_x =
            matern52_kernel(
                x,
                x;
                lengthscale=lengthscale,
                outputscale=outputscale,
            ) -
            dot(kx, Ky \ kx)

        H = zeros(length(x), length(x))

        for i in eachindex(X)
            H .+= α[i] .* matern52_hessian(
                x,
                X[i];
                lengthscale=lengthscale,
                outputscale=outputscale,
            )
        end

        push!(means, mean_x)
        push!(variances, variance_x)
        push!(hessians, H)
    end

    return means, variances, hessians
end


# ---------------------------------------------------------------------------
# Fixed training data
# ---------------------------------------------------------------------------

X = [
    [0.0, 0.0],
    [0.5, 0.2],
    [0.2, 0.8],
    [0.8, 0.7],
]

y = [
    1.0,
    0.4,
    0.7,
    0.2,
]

prior_mean = mean(y)

nugget = 1e-9
outputscale = 2.3

query_points = [
    # Generic point.
    [0.3, 0.4],

    # Exact GP training point. Important for Hessian regression.
    [0.5, 0.2],

    # Another generic point.
    [0.9, 0.3],
]


# ---------------------------------------------------------------------------
# Isotropic reference
# ---------------------------------------------------------------------------

isotropic_lengthscale = [0.8, 0.8]

iso_mean, iso_variance, iso_hessian =
    build_gp_reference(
        X,
        y,
        query_points;
        prior_mean=prior_mean,
        lengthscale=isotropic_lengthscale,
        outputscale=outputscale,
        nugget=nugget,
    )


# ---------------------------------------------------------------------------
# ARD reference
# ---------------------------------------------------------------------------

ard_lengthscale = [0.7, 1.4]

ard_mean, ard_variance, ard_hessian =
    build_gp_reference(
        X,
        y,
        query_points;
        prior_mean=prior_mean,
        lengthscale=ard_lengthscale,
        outputscale=outputscale,
        nugget=nugget,
    )


# ---------------------------------------------------------------------------
# Write TOML
# ---------------------------------------------------------------------------

function matrix_to_rows(H)
    return [
        collect(row)
        for row in eachrow(H)
    ]
end


reference = Dict(
    "training_data" => Dict(
        "X" => X,
        "Y" => y,
    ),

    "hyperparameters" => Dict(
        "prior_mean" => prior_mean,
        "nugget" => nugget,
        "outputscale" => outputscale,
    ),

    "query_points" => query_points,

    "isotropic" => Dict(
        "lengthscale" => isotropic_lengthscale,
        "posterior_mean" => iso_mean,
        "posterior_variance" => iso_variance,
        "posterior_hessian" => [
            matrix_to_rows(H)
            for H in iso_hessian
        ],
    ),

    "ard" => Dict(
        "lengthscale" => ard_lengthscale,
        "posterior_mean" => ard_mean,
        "posterior_variance" => ard_variance,
        "posterior_hessian" => [
            matrix_to_rows(H)
            for H in ard_hessian
        ],
    ),
)


output_path = joinpath(
    @__DIR__,
    "..",
    "data",
    "gp_reference.toml",
)

open(output_path, "w") do io
    TOML.print(io, reference)
end

println("Wrote GP reference data to:")
println(output_path)