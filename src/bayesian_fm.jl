"""
    BayesianFM(f::Matrix{Float64}, R::Matrix{Float64}, sim_length::Int)

Perform Bayesian Fama-MacBeth regression.
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

    Threads.@threads for i in 1:sim_length
        # Draw from inverse Wishart
        Sigma = rand(iw_dist)
        
        # Extract components (matching R's indexing approach)
        Sigma_R = Sigma[k+1:end, k+1:end]
        Sigma_f = Sigma[1:k, 1:k]
        Sigma_Rf = Sigma[k+1:end, 1:k]

        # Draw means (matching R's approach)
        Var_mu_half = cholesky(Sigma/t).U'
        mu = mu_ols + Var_mu_half * randn(size(Y, 2))
        
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