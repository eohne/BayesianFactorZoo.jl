"""
    BayesianFM(f::Matrix{Float64}, R::Matrix{Float64}, sim_length::Int)

Bayesian Fama-MacBeth regression. Similar to BayesianSDF but estimates factors' risk premia rather than risk prices.

# Arguments 
- `f`: Matrix of factors with dimension ``t \\times k``, where ``k`` is the number of factors and ``t`` is the number of periods
- `R`: Matrix of test assets with dimension ``t \\times N``, where ``t`` is the number of periods and ``N`` is the number of test assets
- `sim_length`: Length of MCMCs

# Details
Unlike BayesianSDF, we use factor loadings, ``\\beta_f``, instead of covariance exposures, ``C_f``, in the Fama-MacBeth regression. After obtaining posterior draws of ``\\mu_Y`` and ``\\Sigma_Y`` (see BayesianSDF), we calculate:


# Returns
Returns a BayesianFMOutput struct containing:
- `lambda_ols_path::Matrix{Float64}`: Matrix of size sim_length × (k+1) containing OLS risk premia estimates. First column is ``\\lambda_c`` for constant term, next k columns are factor risk premia.
- `lambda_gls_path::Matrix{Float64}`: Matrix of size sim_length × (k+1) containing GLS risk premia estimates.
- `R2_ols_path::Vector{Float64}`: Vector of length sim_length containing OLS ``R^2`` draws.
- `R2_gls_path::Vector{Float64}`: Vector of length sim_length containing GLS ``R^2`` draws.
- Metadata fields accessible via dot notation:
 - `n_factors::Int`: Number of factors (k)
 - `n_assets::Int`: Number of test assets (N)
 - `n_observations::Int`: Number of time periods (t)
 - `sim_length::Int`: Number of MCMC iterations performed


# References
Bryzgalova S, Huang J, Julliard C (2023). "Bayesian solutions for the factor zoo: We just ran two quadrillion models." Journal of Finance, 78(1), 487–557.

# Examples
```julia
# Run Bayesian FM regression with 10,000 iterations  
results = BayesianFM(f, R, 10_000)

# Access results
ols_risk_premia = mean(results.lambda_ols_path, dims=1)  # Mean OLS risk premia
gls_r2 = mean(results.R2_gls_path)  # Mean GLS R²
```
"""
function BayesianFM(f::Matrix{Float64}, R::Matrix{Float64}, sim_length::Int)

    t, k = size(f)
    N = size(R, 2)
    Y = hcat(f, R)
    check_input2(f,R);

    # Initialize storage
    lambda_ols_path = zeros(sim_length, k + 1)
    lambda_gls_path = zeros(sim_length, k + 1)
    R2_ols_path = zeros(sim_length)
    R2_gls_path = zeros(sim_length)

    # Calculate initial estimates (matching R)
    Sigma_ols = cov(Y)
    mu_ols = reshape(mean(Y, dims=1)', :, 1)
    
    # Setup inverse Wishart with exact R scale
    iw_dist = InverseWishart(t - 1, t * Sigma_ols)
    rngs = [MersenneTwister(i) for i in 1:sim_length]
    Threads.@threads for i in 1:sim_length
        # Draw from inverse Wishart
        mtwist = rngs[i]
        Sigma = rand(mtwist,iw_dist)
        
        # Extract components (matching R's indexing approach)
        Sigma_R = Sigma[k+1:end, k+1:end]
        Sigma_f = Sigma[1:k, 1:k]
        Sigma_Rf = Sigma[k+1:end, 1:k]

        # Draw means (matching R's approach)
        Var_mu_half = cholesky(Sigma/t).U'
        mu = mu_ols + Var_mu_half * randn(mtwist,size(Y, 2))
        
        # Extract means
        a = mu[k+1:end]
        mu_f = mu[1:k]

        # Calculate C (beta) using R's covariance approach
        C = Sigma_Rf * inv(Sigma_f)
        Sigma_eps = Sigma_R - Sigma_Rf * inv(Sigma_f) * Sigma_Rf'

        # Build H matrix
        H = hcat(ones(N), C)
        
        # OLS estimation (matching R)
        HH_inv = inv(H'H)
        Lambda_ols = HH_inv * (H'a)
        
        # GLS estimation (matching R)
        Sigma_eps_inv = inv(Sigma_eps)
        Lambda_gls = inv(H'Sigma_eps_inv * H) * (H'Sigma_eps_inv * a)

        # Calculate R² (matching R's formulas exactly)
        R2_ols = 1 - ((a - H * Lambda_ols)'*(a - H * Lambda_ols))[1] / ((N-1)*var(vec(a)))
        R2_ols_adj = 1 - (1-R2_ols) * (N-1) / (N-1-k)
        
        # GLS R² (matching R)
        a_centered = a .- mean(a)
        R2_gls = 1 - (a - H * Lambda_gls)'Sigma_eps_inv*(a - H * Lambda_gls) / 
                     (a_centered'Sigma_eps_inv*a_centered)
        R2_gls_adj = 1 - (1-R2_gls) * (N-1) / (N-1-k)

        # Store results
        lambda_ols_path[i, :] = vec(Lambda_ols)
        lambda_gls_path[i, :] = vec(Lambda_gls)
        R2_ols_path[i] = R2_ols_adj
        R2_gls_path[i] = R2_gls_adj
    end

    return BayesianFMOutput(
    lambda_ols_path,
    lambda_gls_path, 
    R2_ols_path,
    R2_gls_path;
    n_factors=size(f, 2),
    n_assets=size(R, 2),
    n_observations=size(R, 1),
    sim_length=sim_length
    )
end