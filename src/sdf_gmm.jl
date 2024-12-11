"""
    SDF_gmm(R::Matrix{Float64}, f::Matrix{Float64}, W::Matrix{Float64})

GMM estimation of factor risk prices under linear SDF framework.

# Arguments
- `R`: Matrix of test assets with dimension ``t \\times N``
- `f`: Matrix of factors with dimension ``t \\times k``
- `W`: Weighting matrix for GMM estimation, dimension ``(N+k) \\times (N+k)``

# Returns 
Returns a SDFGMMOutput struct containing:
- `lambda_gmm::Vector{Float64}`: Vector of length k+1 containing risk price estimates (includes intercept).
- `mu_f::Vector{Float64}`: Vector of length k containing estimated factor means.
- `Avar_hat::Matrix{Float64}`: Matrix of size (2k+1) × (2k+1) containing asymptotic covariance matrix.
- `R2_adj::Float64`: Adjusted cross-sectional ``R^2``.
- `S_hat::Matrix{Float64}`: Matrix of size (N+k) × (N+k) containing estimated spectral density matrix.
- Metadata fields accessible via dot notation:
 - `n_factors::Int`: Number of factors (k)
 - `n_assets::Int`: Number of test assets (N)
 - `n_observations::Int`: Number of time periods (t)

# Notes
- Input matrices R and f must have the same number of rows (time periods)
- The weighting matrix W should match dimensions (N+k) × (N+k)
- For tradable factors, weighting matrix should impose self-pricing restrictions
- Implementation assumes no serial correlation in moment conditions
- R² is adjusted for degrees of freedom
- Standard errors are derived under the assumption of correct specification

# References
Bryzgalova S, Huang J, Julliard C (2023). "Bayesian solutions for the factor zoo: We just ran two quadrillion models." Journal of Finance, 78(1), 487–557.

Hansen, Lars Peter (1982). "Large Sample Properties of Generalized Method of Moments Estimators." Econometrica, 50(4), 1029-1054.

# Examples
```julia
# Construct OLS weighting matrix
W_ols = construct_weight_matrix(R, f, "OLS")

# Perform OLS estimation
results_ols = SDF_gmm(R, f, W_ols)

# Construct GLS weighting matrix
W_gls = construct_weight_matrix(R, f, "GLS")

# Perform GLS estimation
results_gls = SDF_gmm(R, f, W_gls)

# Access results
risk_prices = results_ols.lambda_gmm[2:end]  # Factor risk prices (excluding intercept)
std_errors = sqrt.(diag(results_ols.Avar_hat)[2:end])  # Standard errors
r_squared = results_ols.R2_adj  # Adjusted R²
```

# See Also
- `construct_weight_matrix`: Function to construct appropriate OLS/GLS weighting matrices
- `BayesianSDF`: Bayesian alternative that is robust to weak factors
"""
function SDF_gmm(R::Matrix{Float64}, f::Matrix{Float64}, W::Matrix{Float64})
    # Get dimensions (matching R variable names)
    T1 = size(R, 1)
    N = size(R, 2)
    K = size(f, 2)

    # Calculate basic statistics exactly as R
    C_f = cov(R, f)
    one_N = ones(N, 1)
    one_K = ones(K, 1)
    one_T = ones(T1, 1)
    C = hcat(one_N, C_f)
    mu_R = mean(R, dims=1)'
    mu_f = mean(f, dims=1)'

    # GMM estimation exactly as R
    W1 = W[1:N, 1:N]
    lambda_gmm = inv(transpose(C)*W1*C) * (transpose(C)*W1*mu_R)
    lambda_c = lambda_gmm[1]
    lambda_f = lambda_gmm[2:end]

    # Estimate spectral matrix exactly as R
    f_demean = f .- one_T * transpose(mu_f)
    moments = zeros(T1, N+K)
    moments[:, N+1:end] = f_demean

    for t in 1:T1
        R_t = R[t:t, :]'
        f_t = f[t:t, :]'
        moments[t, 1:N] = transpose(R_t .- lambda_c*one_N .- R_t*(transpose(f_t.-mu_f)*lambda_f))
    end
    S_hat = cov(moments)

    # Estimate asymptotic variance exactly as R
    G_hat = zeros(N+K, 2K+1)
    G_hat[1:N, 1] .= -1
    G_hat[1:N, 2:K+1] = -C_f
    G_hat[1:N, K+2:end] = mu_R * transpose(lambda_f)
    G_hat[N+1:end, K+2:end] = -Matrix(I, K, K)

    Avar_hat = (1/T1) * (
        inv(transpose(G_hat)*W*G_hat) * 
        transpose(G_hat)*W*S_hat*W*G_hat *
        inv(transpose(G_hat)*W*G_hat)
    )

    # Calculate R² exactly as R
    R2 = (1 .- transpose(mu_R .- C*lambda_gmm) * W1 * (mu_R .- C*lambda_gmm) ./
             (transpose(mu_R .- mean(mu_R))*W1*(mu_R .- mean(mu_R))))[1]
    R2_adj = 1 - (1-R2) * (N-1) / (N-1-K)

    # At the end of the SDF_gmm function, convert lambda_gmm and mu_f to vectors
    return SDFGMMOutput(
        vec(lambda_gmm),         # Convert to vector
        vec(transpose(mu_f)),              # Convert to vector
        Avar_hat,
        R2_adj,
        S_hat;
        n_factors=K,
        n_assets=N,
        n_observations=T1
    )
end



"""
    construct_weight_matrix(R::Matrix{Float64}, f::Matrix{Float64}, 
                         type::String="OLS", kappa::Float64=1e6)

Construct weighting matrix for GMM estimation of linear SDF models.

# Arguments
- R: Matrix of test assets with dimension ``t \\times N``
- f: Matrix of factors with dimension ``t \\times k``
- type: "OLS" or "GLS", default="OLS"
- kappa: Large constant for factor moment conditions, default=1e6

# Details
Constructs a ``(N+k) \\times (N+k)`` block diagonal weighting matrix W:

```math
W = \\begin{bmatrix} W_R & 0_{N\\times k} \\\\ 0_{k\\times N} & \\kappa I_k \\end{bmatrix}
```

where:

OLS (type="OLS"):

``W_R = I_N``

GLS (type="GLS"):

``W_R = \\Sigma_R^{-1}``

The structure reflects GMM moment conditions:

```math
E[g_t(\\lambda_c,\\lambda_f,\\mu_f)] = E[(R_t - \\lambda_c1_N - R_t(f_t - \\mu_f)'\\lambda_f), (f_t - \\mu_f)] = [0_N, 0_k]
```

# Returns
Returns a Matrix{Float64} of size ``(N+k) \\times (N+k)`` containing the weighting matrix W with structure:
- Upper-left block: Identity (OLS) or inverse return covariance (GLS)
- Lower-right block: ``\\kappa I_k``
- Off-diagonal blocks: Zero matrices

Note: The returned matrix matches the dimension requirements of SDF_gmm function

# Notes
- Input matrices R and f must have the same number of rows (time periods)
- The GLS version requires a well-conditioned return covariance matrix
- κ should be large enough to ensure accurate factor mean estimation
- Output matches dimensions required by SDF_gmm function
- Block structure is optimal under conditional homoskedasticity

# References
Bryzgalova S, Huang J, Julliard C (2023). "Bayesian solutions for the factor zoo: We just ran two quadrillion models." Journal of Finance, 78(1), 487–557.

Hansen, Lars Peter (1982). "Large Sample Properties of Generalized Method of Moments Estimators." Econometrica, 50(4), 1029-1054.

# Examples
```julia
# Construct OLS weighting matrix
W_ols = construct_weight_matrix(R, f, "OLS")

# Construct GLS weighting matrix
W_gls = construct_weight_matrix(R, f, "GLS")

# Use custom kappa value
W_custom = construct_weight_matrix(R, f, "OLS", 1e8)

# Use in GMM estimation
results_ols = SDF_gmm(R, f, W_ols)
results_gls = SDF_gmm(R, f, W_gls)
```

# See Also 
- `SDF_gmm`: Main function using these weighting matrices
"""
function construct_weight_matrix(R::Matrix{Float64}, f::Matrix{Float64}, type::String="OLS", kappa::Float64=1e6)
    N = size(R, 2)  # number of test assets
    k = size(f, 2)  # number of factors
    total_dim = N + k  # total dimension needed for the weight matrix

    if type == "OLS"
        W_R = I(N)  # identity matrix for OLS
    else  # GLS
        Sigma_R = cov(R)
        W_R = inv(Sigma_R)  # inverse covariance matrix for GLS
    end

    # Construct full weight matrix with factor moments
    W = zeros(total_dim, total_dim)
    W[1:N, 1:N] = W_R
    W[N+1:end, N+1:end] = kappa * I(k)

    return W
end