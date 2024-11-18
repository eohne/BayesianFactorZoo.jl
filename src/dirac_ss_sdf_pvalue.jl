"""
    dirac_ss_sdf_pvalue(f::Matrix{Float64}, R::Matrix{Float64}, sim_length::Int, 
                       lambda0::Vector{Float64}; kwargs...)

Hypothesis testing for risk prices using Dirac spike-and-slab prior.
"""
function dirac_ss_sdf_pvalue(f::Matrix{Float64}, R::Matrix{Float64}, sim_length::Int,
    lambda0::Vector{Float64}; psi0::Float64=1.0,
    max_k::Union{Int,Nothing}=nothing)

    # Get dimensions
    t, k = size(f)
    N = size(R, 2)
    p = k + N

    # Set max_k if not provided
    if isnothing(max_k)
        max_k = k
    end

    # Check input validity
    check_input2(f, R)

    # Combine data
    Y = hcat(f, R)
    Sigma_ols = cov(Y)
    Corr_ols = cor(Y)
    sd_ols = vec(std(Y, dims=1))
    mu_ols = vec(mean(Y, dims=1))

    # Initialize storage
    lambda_path = zeros(sim_length, k + 1)
    gamma_path = zeros(sim_length, k)

    # Set prior distribution for lambda_f (match R exactly)
    rho = cor(Y)[(k+1):p, 1:k]
    rho_demean = rho .- repeat(mean(rho, dims=1), N)
    
    # Calculate psi (match R)
    if k == 1
        psi = psi0 * (transpose(rho_demean) * rho_demean)[1]
    else
        psi = psi0 * diag(transpose(rho_demean) * rho_demean)
    end
    
    # Setup D matrix exactly as R
    D = Diagonal(vcat([1/100000], 1 ./ psi))

    # Generate modelsets exactly as R
    modelsets = [zeros(Int, k)]  # null model
    for l in 1:max_k
        for combo in combinations(1:k, l)
            model = zeros(Int, k)
            model[combo] .= 1
            push!(modelsets, model)
        end
    end
    model_num = length(modelsets)

    # MCMC loop
    Threads.@threads for j in 1:sim_length
        # First stage: time series regression
        Sigma = rand(InverseWishart(t-1, t * Sigma_ols))
        Var_mu_half = cholesky(Sigma/t).U
        mu = mu_ols + transpose(Var_mu_half) * randn(p)

        # Calculate standardized quantities
        sd_Y = sqrt.(diag(Sigma))
        corr_Y = Sigma ./ (sd_Y * transpose(sd_Y))
        C_f = corr_Y[(k+1):p, 1:k]
        a = mu[(k+1):p] ./ sd_Y[(k+1):p]

        # Second stage: cross-sectional regression
        beta_f = C_f
        beta = hcat(ones(N), C_f)
        log_prob = zeros(model_num)

        # Calculate model probabilities
        for i in 1:model_num
            if i == 1  # null model
                H_i = ones(N, 1)
                D_i = [1/100000;;]  # 1x1 matrix
                a_gamma = a - beta_f * lambda0
            else
                index = findall(x -> x == 1, modelsets[i])
                H_i = hcat(ones(N), beta_f[:, index])
                D_i = D[vcat(1, index .+ 1), vcat(1, index .+ 1)]
                
                if length(index) == k  # full model
                    a_gamma = a
                else
                    not_index = setdiff(1:k, index)
                    a_gamma = a - beta_f[:, not_index] * lambda0[not_index]
                end
            end

            # Calculate lambda_i and SSR exactly as R
            HH_D_inv = inv(cholesky(transpose(H_i)*H_i + D_i))
            lambda_i = HH_D_inv * (transpose(H_i)*a_gamma)
            SSR_i = (transpose(a_gamma)*a_gamma - 
                    transpose(a_gamma)*H_i * HH_D_inv * transpose(H_i)*a_gamma)[1]
            
            # Log probability calculation matching R
            log_prob[i] = 0.5 * (logdet(D_i) - logdet(transpose(H_i)*H_i + D_i)) - 
                         0.5*N*log(0.5*SSR_i)
        end

        # Draw model
        probs = exp.(log_prob)
        probs = probs ./ sum(probs)
        i = rand(Categorical(probs))

        # Handle drawn model
        if i == 1  # null model
            index = Int[]
            H_i = ones(N, 1)
            D_i = [1/100000;;]
            a_gamma = a - beta_f * lambda0
        else
            index = findall(x -> x == 1, modelsets[i])
            H_i = hcat(ones(N), beta_f[:, index])
            D_i = D[vcat(1, index .+ 1), vcat(1, index .+ 1)]
            gamma_path[j, index] .= 1
            
            if length(index) == k
                a_gamma = a
            else
                not_index = setdiff(1:k, index)
                a_gamma = a - beta_f[:, not_index] * lambda0[not_index]
            end
        end

        # Draw sigma2 and lambda
        HH_D_inv = inv(cholesky(transpose(H_i)*H_i + D_i))
        Lambda_hat = HH_D_inv * (transpose(H_i)*a_gamma)
        sigma2 = rand(InverseGamma(N/2, 
                    (transpose(a_gamma - H_i*Lambda_hat)*(a_gamma - H_i*Lambda_hat))[1]/2))
        
        cov_Lambda = sigma2 * HH_D_inv
        Lambda = Lambda_hat + transpose(cholesky(cov_Lambda).U) * randn(length(Lambda_hat))

        # Store results exactly as R
        if length(index) == 0  # null model
            lambda_path[j, 1] = Lambda[1]
            lambda_path[j, 2:end] = lambda0
        else
            lambda_path[j, vcat(1, index .+ 1)] = Lambda
            unselected = setdiff(1:k, index)
            if !isempty(unselected)
                lambda_path[j, unselected .+ 1] = lambda0[unselected]
            end
        end
    end

    # Compute model probabilities
    model_probs = zeros(model_num, k + 1)
    for (i, model) in enumerate(modelsets)
        model_probs[i, 1:k] = model
        model_probs[i, k+1] = mean(all(gamma_path .== model', dims=2))
    end

    return DiracSSSDFOutput(
        gamma_path,
        lambda_path,
        model_probs;
        n_factors=k,
        n_assets=N,
        n_observations=t,
        sim_length=sim_length
    )
end