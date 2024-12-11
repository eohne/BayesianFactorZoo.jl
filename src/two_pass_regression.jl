"""
    TwoPassRegression(f::Matrix{Float64}, R::Matrix{Float64})

Classical Fama-MacBeth two-pass regression.

# Arguments
- f: Matrix of factors with dimension ``t \\times k``
- R: Matrix of test assets with dimension ``t \\times N``

# Returns
Returns a TwoPassRegressionOutput struct containing:
- lambda::Vector{Float64}: Vector of length k+1 containing OLS risk premia estimates (includes intercept).
- lambda_gls::Vector{Float64}: Vector of length k+1 containing GLS risk premia estimates.
- t_stat::Vector{Float64}: Vector of length k+1 containing OLS t-statistics.
- t_stat_gls::Vector{Float64}: Vector of length k+1 containing GLS t-statistics.
- R2_adj::Float64: OLS adjusted R².
- R2_adj_GLS::Float64: GLS adjusted R².
- alpha::Vector{Float64}: Vector of length N containing OLS pricing errors.
- t_alpha::Vector{Float64}: Vector of length N containing t-statistics for OLS pricing errors.
- beta::Matrix{Float64}: Matrix of size N × k containing factor loadings.
- cov_epsilon::Matrix{Float64}: Matrix of size N × N containing residual covariance.
- cov_lambda::Matrix{Float64}: Matrix of size (k+1) × (k+1) containing OLS covariance matrix of risk premia.
- cov_lambda_gls::Matrix{Float64}: Matrix of size (k+1) × (k+1) containing GLS covariance matrix of risk premia.
- R2_GLS::Float64: Unadjusted GLS R².
- cov_beta::Matrix{Float64}: Matrix of size (N(k+1)) × (N(k+1)) containing covariance matrix of beta estimates.
- Metadata fields accessible via dot notation:
 - n_factors::Int: Number of factors (k)
 - n_assets::Int: Number of test assets (N)
 - n_observations::Int: Number of time periods (t)

# Notes
- Input matrices f and R must have the same number of rows (time periods)
- The method is vulnerable to bias from weak and useless factors
- Standard errors account for the EIV problem but assume serial independence
- Both OLS and GLS estimates are computed with appropriate standard errors
- R² values are adjusted for degrees of freedom
- Includes corrections for using factors as test assets when applicable

# References
Fama, Eugene F., and James D. MacBeth, 1973, Risk, return, and equilibrium: Empirical tests, Journal of Political Economy 81, 607-636.

Shanken, Jay, 1992, On the estimation of beta-pricing models, Review of Financial Studies 5, 1-33.

# Examples
```julia
# Perform two-pass regression
results = TwoPassRegression(f, R)

# Access OLS results
risk_premia = results.lambda[2:end]  # Factor risk premia (excluding intercept)
t_stats = results.t_stat[2:end]      # t-statistics
r2_ols = results.R2_adj              # Adjusted R²
pricing_errors = results.alpha        # Pricing errors

# Access GLS results
risk_premia_gls = results.lambda_gls[2:end]  
t_stats_gls = results.t_stat_gls[2:end]
r2_gls = results.R2_adj_GLS

# First-pass results
betas = results.beta                  # Factor loadings
std_errors_beta = sqrt.(diag(results.cov_beta))  # Standard errors for betas
```

# See Also
- `BayesianFM`: Bayesian version that is robust to weak factors
- `SDF_gmm`: GMM-based alternative estimation approach
"""
function TwoPassRegression(f::Matrix{Float64}, R::Matrix{Float64})
    t = size(R, 1)
    k = size(f, 2)
    N = size(R, 2)

    # Center the factors
    ET_f = transpose(mean(f, dims=1))
    f_centered = f .- transpose(ET_f)

    # Step 1: Time-Series Regression
    X = hcat(ones(t), f_centered)
    XX_inv = inv(transpose(X) * X)
    B = (transpose(R) * X) * XX_inv
    beta = B[:, 2:k+1]  # factor loadings
    epsilon = transpose(R) - B * transpose(X)  # error terms
    cov_epsilon = (epsilon * transpose(epsilon)) / t  # error covariance matrix
    cov_beta = kron(cov_epsilon, XX_inv)

    # Step 2: Cross-Sectional Regression (OLS)
    C = hcat(ones(N), beta) 
    mu = transpose(R) * (ones(t)./t)
    lambda = inv(transpose(C) * C) * (transpose(C) * mu)
    alpha = mu - C * lambda

    # Calculate asymptotic covariance matrix
    Omega_F = zeros(k + 1, k + 1)
    Omega_F_inv = zeros(k + 1, k + 1)
    Omega_F[2:k+1, 2:k+1] = cov(f)
    Omega_F_inv[2:k+1, 2:k+1] = inv(cov(f))

    # Calculate lambda covariance
    term1 = inv(transpose(C) * C) * transpose(C) * cov_epsilon * C * inv(transpose(C) * C)
    term2 = (1 + (transpose(lambda)*Omega_F_inv*lambda))
    cov_lambda = (1 / t) * (term1 * term2 + Omega_F)

    tstat = vec(lambda) ./ sqrt.(diag(cov_lambda))

    # Calculate alpha covariance
    y = I(N) - C * inv(transpose(C) * C) * transpose(C)
    cov_alpha = (1 / t) * y * cov_epsilon * y * (1 + (transpose(lambda)*Omega_F_inv*lambda))
    t_alpha = vec(alpha) ./ sqrt.(diag(cov_alpha))

    # Step 3: Cross-Sectional Regression (GLS)
    lambda_gls = inv(transpose(C) * inv(cov_epsilon) * C) * (transpose(C) * inv(cov_epsilon) * mu)
    cov_lambda_gls = (1 / t) * (
        inv(transpose(C) * inv(cov_epsilon) * C) * (1 + (transpose(lambda)*Omega_F_inv*lambda)) +
        Omega_F
    )
    tstat_gls = vec(lambda_gls) ./ sqrt.(diag(cov_lambda_gls))

    # Calculate R²
    # OLS R²
    alpha = mu - C * lambda
    TSS = transpose(mu .- mean(mu)) * (mu .- mean(mu))
    R2 = 1 - (transpose(alpha)*alpha) / TSS
    R2_adj = 1 - (1 - R2) * (N - 1) / (N - 1 - k)

    # GLS R²
    alpha_GLS = mu - C * lambda_gls
    R2_GLS = 1 - (transpose(alpha_GLS)*inv(cov_epsilon)*alpha_GLS) /
                 (transpose(mu .- mean(mu))*inv(cov_epsilon)*(mu.-mean(mu)))
    R2_adj_GLS = 1 - (1 - R2_GLS) * (N - 1) / (N - 1 - k)

    return TwoPassRegressionOutput(  
    lambda,
    lambda_gls,
    tstat,
    tstat_gls,
    R2_adj,
    R2_adj_GLS,
    alpha,
    t_alpha,
    beta,
    cov_epsilon,
    cov_lambda,
    cov_lambda_gls,
    R2_GLS,
    cov_beta;
    n_factors=k,
    n_assets=N,
    n_observations=t
    )
end