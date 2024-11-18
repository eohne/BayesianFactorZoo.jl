"""
    SDF_gmm(R::Matrix{Float64}, f::Matrix{Float64}, W::Matrix{Float64})

GMM estimation of factors' risk prices under linear SDF framework.

# Arguments
- `R`: Matrix of test assets with dimension t × N
- `f`: Matrix of factors with dimension t × k
- `W`: Weighting matrix in GMM estimation. For OLS use identity matrix,
       for GLS use inverse of return covariance matrix

# Returns
Named tuple containing:
- `lambda_gmm`: GMM estimates of risk prices
- `mu_f`: Sample means of factors
- `Avar_hat`: Asymptotic covariance matrix 
- `R2_adj`: Adjusted cross-sectional R²
- `S_hat`: Spectral matrix
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
    construct_weight_matrix(R::Matrix{Float64}, f::Matrix{Float64}, type::String="OLS", kappa::Float64=1e6)

Construct the weighting matrix for GMM estimation.

# Arguments
- `R`: Matrix of test assets with dimension t × N
- `f`: Matrix of factors with dimension t × k
- `type`: "OLS" or "GLS" (default: "OLS")
- `kappa`: Large constant for factor moment conditions (default: 1e6)

# Returns
- Weight matrix W for GMM estimation with dimension (N+k) × (N+k), where:
  - Top-left block is N × N identity (OLS) or inverse covariance (GLS)
  - Bottom-right block is k × k scaled identity
  - Off-diagonal blocks are zero
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