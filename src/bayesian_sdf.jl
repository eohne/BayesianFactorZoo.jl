"""
    BayesianSDF(f::Matrix{Float64}, R::Matrix{Float64}, sim_length::Int=10000;
                intercept::Bool=true, type::String="OLS", prior::String="Flat",
                psi0::Float64=5.0, d::Float64=0.5)

Bayesian estimation of Linear SDF (B-SDF)
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

    # MCMC loop
    Threads.@threads for i in 1:sim_length
        # First stage: time series regression
        Sigma = rand(iw_dist)
        Sigma_R = Sigma[k+1:end, k+1:end]
        
        # Draw means (matching R's approach)
        Var_mu_half = cholesky(Sigma/t).U
        mu = mu_ols + transpose(Var_mu_half) * randn(p)
        
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