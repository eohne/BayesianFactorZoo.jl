"""
    BayesianSDF(f::Matrix{Float64}, R::Matrix{Float64}, sim_length::Int=10000; 
               intercept::Bool=true, type::String="OLS", prior::String="Flat",
               psi0::Float64=5.0, d::Float64=0.5)

Bayesian estimation of Linear SDF (B-SDF).

# Arguments
- `f`: Matrix of factors with dimension ``t \\times k``
- `R`: Matrix of test assets with dimension ``t \\times N``
- `sim_length`: Length of MCMCs
- `intercept`: Include intercept if true, default=true
- `type`: "OLS" or "GLS", default="OLS"
- `prior`: "Flat" or "Normal", default="Flat"
- `psi0`: Hyperparameter for normal prior, default=5
- `d`: Hyperparameter for normal prior, default=0.5

# Returns
Returns a BayesianSDFOutput struct containing:
- `lambda_path::Matrix{Float64}`: Matrix of size sim_length × (k+1) if intercept=true, or sim_length × k if false. Contains posterior draws of risk prices.
- `R2_path::Vector{Float64}`: Vector of length sim_length containing ``R^2`` draws.
- Metadata fields accessible via dot notation:
 - `n_factors::Int`: Number of factors (k)
 - `n_assets::Int`: Number of test assets (N)
 - `n_observations::Int`: Number of time periods (t)
 - `sim_length::Int`: Number of MCMC iterations performed
 - `prior::String`: Prior specification used ("Flat" or "Normal")
 - `estimation_type::String`: Estimation type used ("OLS" or "GLS")

# Notes
- Input matrices f and R must have the same number of rows (time periods)
- Number of test assets (N) must be larger than number of factors (k) when including intercept
- Number of test assets (N) must be >= number of factors (k) when excluding intercept
- The function performs no pre-standardization of inputs
- Risk prices are estimated in the units of the input data (typically monthly returns)

# References
Bryzgalova S, Huang J, Julliard C (2023). "Bayesian solutions for the factor zoo: We just ran two quadrillion models." Journal of Finance, 78(1), 487–557.

# Examples
```julia
# Basic usage with default settings
results = BayesianSDF(f, R)

# Use GLS with normal prior
results_gls = BayesianSDF(f, R, 10_000; 
                        type="GLS", 
                        prior="Normal",
                        psi0=5.0,
                        d=0.5)

# Access results
risk_prices = mean(results.lambda_path, dims=1)
r2_values = mean(results.R2_path)
```
"""
function BayesianSDF(f::Matrix{Float64}, R::Matrix{Float64}, sim_length::Int=10000;
    intercept::Bool=true, type::String="OLS", prior::String="Flat",
    psi0::Float64=5.0, d::Float64=0.5)


    t, k = size(f)   # factors: t × k
    N = size(R, 2)   # returns: t × N
    p = N + k

    # Check inputs
    check_input(f, R, intercept, type, prior)

    # Combine data (no scaling like in R)
    Y = hcat(f, R)
    Sigma_ols = cov(Y)
    mu_ols = transpose(mean(Y, dims=1))
    ones_N = ones(N, 1)

    # Initialize storage
    lambda_path = zeros(sim_length, intercept ? k + 1 : k)
    R2_path = zeros(sim_length)

    # Set up prior for risk prices
    rho = cor(Y)[k+1:end, 1:k]
    rho_demean = rho .- repeat(mean(rho, dims=1), N)

    # Set up prior matrix D (matching R's logic)
    if prior == "Normal"
        psi = psi0 * diag(transpose(rho_demean) * rho_demean)
        if intercept
            D = Diagonal(vcat([1/100000], 1 ./ psi)) * (1/t)^d
        else
            if k == 1
                D = fill((1/psi[1]) * (1/t)^d, 1, 1)
            else
                D = Diagonal(1 ./ psi) * (1/t)^d
            end
        end
    end

    # Setup inverse Wishart with exact R scale
    iw_dist = InverseWishart(t - 1, t * Sigma_ols)
    rngs = [MersenneTwister(i) for i in 1:sim_length]
    # MCMC loop
    Threads.@threads for i in 1:sim_length
        mtwist = rngs[i]
        # First stage: time series regression
        Sigma = rand(mtwist,iw_dist)
        Sigma_R = Sigma[k+1:end, k+1:end]
        
        # Draw means (matching R's approach)
        Var_mu_half = cholesky(Sigma/t).U
        mu = mu_ols + transpose(Var_mu_half) * randn(mtwist,p)
        
        # Calculate quantities for second stage
        sd_Y = sqrt.(diag(Sigma))
        corr_Y = Sigma ./ (sd_Y * transpose(sd_Y))
        C_f = corr_Y[k+1:end, 1:k]
        a = mu[k+1:end] ./ sd_Y[k+1:end]

        # Cross-sectional regression
        H = intercept ? hcat(ones_N, C_f) : C_f
        
        # Use cholesky for matrix inversion like R
        Sigma_inv = inv(cholesky(Hermitian(corr_Y[k+1:end, k+1:end])))

        # Calculate lambda based on prior and type (matching R's formulas)
        local Lambda, R2
        if prior == "Flat"
            if type == "OLS"
                Lambda = inv(cholesky(Hermitian(transpose(H)*H))) * (transpose(H)*a)
                R2 = 1 - ((transpose(a - H * Lambda) * (a - H * Lambda))[1] / ((N-1) * var(vec(a))))
            else # GLS
                Lambda = inv(cholesky(Hermitian(transpose(H)*Sigma_inv*H))) * (transpose(H)*Sigma_inv*a)
                a_centered = a .- mean(a)
                R2 = 1 - transpose(a - H * Lambda)*Sigma_inv*(a - H * Lambda) / 
                        (transpose(a_centered)*Sigma_inv*a_centered)
            end
        else # Normal prior
            if type == "OLS"
                Lambda = inv(cholesky(Hermitian(transpose(H)*H + D))) * (transpose(H)*a)
                R2 = 1 - ((transpose(a - H*Lambda) * (a - H*Lambda))[1] / ((N-1)*var(vec(a))))
            else # GLS
                Lambda = inv(cholesky(Hermitian(transpose(H)*Sigma_inv*H + D))) * (transpose(H)*Sigma_inv*a)
                a_centered = a .- mean(a)
                R2 = 1 - transpose(a - H*Lambda)*Sigma_inv*(a - H*Lambda) / 
                        (transpose(a_centered)*Sigma_inv*a_centered)
            end
        end

        R2 = Float64(R2[1])  # Convert to scalar

        # Store results (no scaling of lambda)
        lambda_path[i, :] = vec(Lambda)
        R2_path[i] = 1 - (1 - R2) * (N - 1) / (N - 1 - k)
    end

    return BayesianSDFOutput(
    lambda_path,
    R2_path;
    n_factors=size(f, 2),
    n_assets=size(R, 2),
    n_observations=size(R, 1),
    sim_length=sim_length,
    prior=prior,
    estimation_type=type
    )
end