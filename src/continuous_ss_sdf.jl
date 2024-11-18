"""
    continuous_ss_sdf(f::Matrix{Float64}, R::Matrix{Float64}, sim_length::Int; 
                     psi0::Float64=1.0, r::Float64=0.001, 
                     aw::Float64=1.0, bw::Float64=1.0, 
                     type::String="OLS", intercept::Bool=true)

SDF model selection with continuous spike-and-slab prior.

# Arguments
- `f`: Matrix of factors
- `R`: Matrix of test assets
- `sim_length`: Length of MCMC chains
- `psi0`: Hyper-parameter for prior (default: 1.0)
- `r`: Hyper-parameter for spike component (default: 0.001)
- `aw`, `bw`: Beta prior parameters for ω (default: 1.0)
- `type`: "OLS" or "GLS" (default: "OLS")
- `intercept`: Whether to include intercept (default: true)

# Returns
Named tuple containing:
- `gamma_path`: Matrix of model inclusion indicators
- `lambda_path`: Matrix of risk price draws
- `sdf_path`: Matrix of SDF draws
- `bma_sdf`: Bayesian Model Averaged SDF
"""
function continuous_ss_sdf(f::Matrix{Float64}, R::Matrix{Float64}, sim_length::Int;
    psi0::Float64=1.0, r::Float64=0.001,
    aw::Float64=1.0, bw::Float64=1.0,
    type::String="OLS", intercept::Bool=true)

    # Get dimensions
    t, k = size(f)
    N = size(R, 2)
    p = k + N

    # Combine data
    Y = hcat(f, R)
    Sigma_ols = cov(Y)
    Corr_ols = cor(Y)
    sd_ols = vec(std(Y, dims=1))
    mu_ols = vec(mean(Y, dims=1))

    # Check input validity
    check_input2(f, R)

    # Initialize storage
    lambda_path = zeros(sim_length, intercept ? k + 1 : k)
    gamma_path = zeros(sim_length, k)
    sdf_path = zeros(sim_length, t)

    # Initialize parameters
    if intercept
        beta_ols = hcat(ones(N), Corr_ols[k+1:p, 1:k])
    else
        beta_ols = Corr_ols[k+1:p, 1:k]
    end

    a_ols = mu_ols[k+1:p] ./ sd_ols[k+1:p]
    Lambda_ols = inv(cholesky(Hermitian(transpose(beta_ols)*beta_ols))) * (transpose(beta_ols) * a_ols)
    sigma2 = (transpose(a_ols - beta_ols * Lambda_ols) * (a_ols - beta_ols * Lambda_ols))[1] / N

    omega = fill(0.5, k)
    gamma = rand(Bernoulli(0.5), k)
    r_gamma = ifelse.(gamma .== 1, 1.0, r)

    # Set prior distribution for lambda_f
    rho = cor(Y)[k+1:p, 1:k]
    if intercept
        rho_mean = vec(mean(rho, dims=1))
        rho_demean = rho .- transpose(rho_mean)
    else
        rho_demean = rho
    end

    # Calculate psi
    if k == 1
        psi = psi0 * (transpose(rho_demean) * rho_demean)[1]
    else
        psi = psi0 * diag(transpose(rho_demean) * rho_demean)
    end

    # Setup inverse Wishart distribution
    iw_dist = InverseWishart(t - 1, t * Sigma_ols)

    # MCMC loop
    Threads.@threads for i in 1:sim_length
        # First stage: time series regression
        Sigma = rand(iw_dist)
        Var_mu_half = cholesky(Hermitian(Sigma/t)).U
        mu = mu_ols + transpose(Var_mu_half) * randn(p)

        # Calculate standardized quantities
        sd_Y = sqrt.(diag(Sigma))
        corr_Y = Sigma ./ (sd_Y * transpose(sd_Y))
        C_f = corr_Y[k+1:p, 1:k]
        a = mu[k+1:p] ./ sd_Y[k+1:p]

        # Second stage: cross-sectional regression
        if intercept
            beta = hcat(ones(N), C_f)
        else
            beta = C_f
        end

        corrR = corr_Y[k+1:p, k+1:p]

        # Setup D matrix
        if intercept
            D = Diagonal(vcat([1 / 100000], 1 ./ (r_gamma .* psi)))
        else
            if k == 1
                D = fill(1/(r_gamma[1] * psi[1]), 1, 1)
            else
                D = Diagonal(1 ./ (r_gamma .* psi))
            end
        end

        # Draw lambda
        if type == "OLS"
            beta_D = transpose(beta)*beta + D
            beta_D_inv = inv(cholesky(Hermitian(beta_D)))
            cov_Lambda = beta_D_inv .* sigma2
            Lambda_hat = beta_D_inv * (transpose(beta)*a)
        else # GLS
            corrR_inv = inv(cholesky(Hermitian(corrR)))
            beta_D = transpose(beta)*corrR_inv*beta + D
            beta_D_inv = inv(cholesky(Hermitian(beta_D)))
            cov_Lambda = beta_D_inv .* sigma2
            Lambda_hat = beta_D_inv * (transpose(beta)*corrR_inv*a)
        end

        # Draw Lambda
        Lambda = Lambda_hat + transpose(cholesky(Hermitian(cov_Lambda)).U) * randn(length(Lambda_hat))

        # Draw gamma
        log_odds = if intercept
            log.(omega ./ (1 .- omega)) .+ 0.5 * log(r) .+ 
            0.5 .* Lambda[2:(k+1)] .^ 2 .* (1/r - 1) ./ (sigma2 .* psi)
        else
            log.(omega ./ (1 .- omega)) .+ 0.5 * log(r) .+ 
            0.5 .* vec(Lambda) .^ 2 .* (1/r - 1) ./ (sigma2 .* psi)
        end
        odds = exp.(log_odds)
        odds = min.(odds, 1000.0)
        prob = odds ./ (1 .+ odds)
        gamma = rand.(Bernoulli.(prob))

        r_gamma = ifelse.(gamma .== 1, 1.0, r)
        gamma_path[i, :] = gamma

        # Draw omega
        omega = rand.(Beta.(aw .+ gamma, bw .+ 1 .- gamma))

        # Draw sigma2
        if type == "OLS"
            resid = a - beta * Lambda
            dof = intercept ? (N + k + 1) / 2 : (N + k) / 2
            scale = (transpose(resid)*resid + transpose(Lambda)*D*Lambda)[1] / 2
            sigma2 = rand(InverseGamma(dof, scale))
        else # GLS
            resid = a - beta * Lambda
            dof = intercept ? (N + k + 1) / 2 : (N + k) / 2
            scale = (transpose(resid)*corrR_inv*resid + transpose(Lambda)*D*Lambda)[1] / 2
            sigma2 = rand(InverseGamma(dof, scale))
        end

        # Store lambda
        lambda_path[i, :] = Lambda

        # Calculate SDF (matching R's approach)
        if intercept
            Lambda_f = Lambda[2:end] ./ vec(std(f, dims=1))
        else
            Lambda_f = Lambda ./ vec(std(f, dims=1))
        end

        sdf = 1 .- f * Lambda_f
        sdf = 1 .+ vec(sdf) .- mean(sdf)  # normalize to mean 1
        sdf_path[i, :] = sdf
    end

    # Calculate BMA-SDF
    bma_sdf = vec(mean(sdf_path, dims=1))

    return ContinuousSSSDFOutput(
    gamma_path,
    lambda_path,
    sdf_path,
    bma_sdf;
    n_factors=k,          # k is total number of factors (k1 + k2)
    n_assets=N,           # N is defined in the function
    n_observations=t,     # t is defined in the function
    sim_length=sim_length
    )
end