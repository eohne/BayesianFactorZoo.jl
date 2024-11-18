"""
    continuous_ss_sdf_v2(f1::Matrix{Float64}, f2::Matrix{Float64}, R::Matrix{Float64}, 
                        sim_length::Int; psi0::Float64=1.0, r::Float64=0.001, 
                        aw::Float64=1.0, bw::Float64=1.0, type::String="OLS", 
                        intercept::Bool=true)

SDF model selection with continuous spike-and-slab prior, treating tradable factors as test assets.
"""
function continuous_ss_sdf_v2(f1::Matrix{Float64}, f2::Matrix{Float64}, R::Matrix{Float64},
    sim_length::Int; psi0::Float64=1.0, r::Float64=0.001,
    aw::Float64=1.0, bw::Float64=1.0, type::String="OLS",
    intercept::Bool=true)

    f = hcat(f1, f2)
    k1 = size(f1, 2)
    k2 = size(f2, 2)
    k = k1 + k2
    N = size(R, 2) + k2
    t = size(R, 1)
    p = k1 + N

    Y = hcat(f, R)
    Sigma_ols = cov(Y)
    Corr_ols = cor(Y)
    sd_ols = std(Y, dims=1)[:]
    mu_ols = mean(Y, dims=1)'

    check_input2(f, hcat(R, f2))

    lambda_path = zeros(sim_length, intercept ? k + 1 : k)
    gamma_path = zeros(sim_length, k)
    sdf_path = zeros(sim_length, t)

    if intercept
        beta_ols = hcat(ones(N), Corr_ols[k1+1:p, 1:k])
    else
        beta_ols = Corr_ols[k1+1:p, 1:k]
    end

    a_ols = mu_ols[k1+1:p] ./ sd_ols[k1+1:p]
    Lambda_ols = inv(beta_ols' * beta_ols) * (beta_ols' * a_ols)
    omega = fill(0.5, k)
    gamma = rand(Bernoulli(0.5), k)
    sigma2 = (1 / N) * ((a_ols - beta_ols * Lambda_ols)'*(a_ols-beta_ols*Lambda_ols))[1]
    r_gamma = ifelse.(gamma .== 1, 1.0, r)

    rho = cor(Y)[k1+1:p, 1:k]
    if intercept
        rho_demean = rho .- mean(rho, dims=1)
    else
        rho_demean = rho
    end

    if k == 1
        psi = psi0 * (rho_demean'*rho_demean)[1]
    else
        psi = psi0 * diag(rho_demean' * rho_demean)
    end

    Threads.@threads for i in 1:sim_length
        # First stage: time series regression
        Sigma = rand(InverseWishart(t - 1, t * Sigma_ols))
        Sigma = Symmetric(Sigma)

        Var_mu = Symmetric(Sigma / t)
        mu = mu_ols + cholesky(Var_mu).U' * randn(p)

        sd_Y = sqrt.(diag(Sigma))
        corr_Y = Symmetric(Sigma ./ (sd_Y * sd_Y'))
        C_f = corr_Y[k1+1:p, 1:k]
        a = mu[k1+1:p] ./ sd_Y[k1+1:p]

        # Second stage: cross-sectional regression
        if intercept
            beta = hcat(ones(N), C_f)
        else
            beta = C_f
        end
        corrR = corr_Y[k1+1:p, k1+1:p]

        if intercept
            D = Diagonal(vcat([1 / 100000], 1 ./ (r_gamma .* psi)))
        else
            D = k == 1 ? fill(1 / (r_gamma[1] * psi), 1, 1) : Diagonal(1 ./ (r_gamma .* psi))
        end

        if type == "OLS"
            beta_D_inv = inv(beta' * beta + D)
            cov_Lambda = sigma2 * beta_D_inv
            Lambda_hat = beta_D_inv * (beta' * a)
        else # GLS
            beta_D_inv = inv(beta' * inv(corrR) * beta + D)
            cov_Lambda = sigma2 * beta_D_inv
            Lambda_hat = beta_D_inv * (beta' * inv(corrR) * a)
        end

        if intercept
            Lambda = Lambda_hat + cholesky(Symmetric(cov_Lambda)).U' * randn(k + 1)
        else
            Lambda = Lambda_hat + cholesky(Symmetric(cov_Lambda)).U' * randn(k)
        end

        if intercept
            log_odds = log.(omega ./ (1 .- omega)) .+ 0.5 * log(r) .+
                       0.5 * Lambda[2:end] .^ 2 .* (1 / r .- 1) ./ (sigma2 .* psi)
        else
            log_odds = log.(omega ./ (1 .- omega)) .+ 0.5 * log(r) .+
                       0.5 * Lambda .^ 2 .* (1 / r .- 1) ./ (sigma2 .* psi)
        end

        odds = exp.(clamp.(log_odds, -100, log(1000)))
        prob = odds ./ (1 .+ odds)
        gamma = rand.(Bernoulli.(prob))
        r_gamma = ifelse.(gamma .== 1, 1.0, r)
        gamma_path[i, :] = gamma

        omega = rand.(Beta.(aw .+ gamma, bw .+ 1 .- gamma))

        if type == "OLS"
            if intercept
                sigma2 = rand(InverseGamma((N + k + 1) / 2,
                    ((a - beta * Lambda)'*(a-beta*Lambda)+Lambda'*D*Lambda)[1] / 2))
            else
                sigma2 = rand(InverseGamma((N + k) / 2,
                    ((a - beta * Lambda)'*(a-beta*Lambda)+Lambda'*D*Lambda)[1] / 2))
            end
        else # GLS
            if intercept
                sigma2 = rand(InverseGamma((N + k + 1) / 2,
                    ((a - beta * Lambda)'*inv(corrR)*(a-beta*Lambda)+Lambda'*D*Lambda)[1] / 2))
            else
                sigma2 = rand(InverseGamma((N + k) / 2,
                    ((a - beta * Lambda)'*inv(corrR)*(a-beta*Lambda)+Lambda'*D*Lambda)[1] / 2))
            end
        end

        lambda_path[i, :] = Lambda
        if intercept
            Lambda_f = Lambda[2:end] ./ std(f, dims=1)[:]
        else
            Lambda_f = Lambda ./ std(f, dims=1)[:]
        end

        sdf = 1 .- (f .- mean(f, dims=1)) * Lambda_f
        sdf = 1 .+ sdf .- mean(sdf)
        sdf_path[i, :] = vec(sdf)
    end

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
