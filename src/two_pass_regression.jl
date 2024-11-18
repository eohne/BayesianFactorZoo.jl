"""
    TwoPassRegression(f::Matrix{Float64}, R::Matrix{Float64})

Perform Fama-MacBeth two-pass regression.
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