mutable struct MCMCOutputs
    gamma_path::Matrix{Float64}
    lambda_path::Matrix{Float64}
    sdf_path::Matrix{Float64}
end

struct MCMCConstants
    f::Matrix{Float64}
    psi::Vector{Float64}
    r::Float64
    aw::Float64
    bw::Float64
    type::String
    intercept::Bool
    rngs::MersenneTwister
    t::Int64
    N::Int64
    k::Int64
    p::Int64
    Y::Matrix{Float64}
    iw_dist::InverseWishart
    mu_ols::Vector{Float64}
    dof2::Float64
end

mutable struct MCMCStates
    r_gamma::Vector{Float64} 
    sigma2::Float64
    omega::Vector{Float64}
end

mutable struct MCMCTemps
    Sigma::Matrix{Float64}
    Var_mu_half::Matrix{Float64}
    mu::Vector{Float64}
    sd_Y::Vector{Float64}
    corr_Y::Matrix{Float64}
    corrR::Matrix{Float64}
    beta::Matrix{Float64}
    a::Vector{Float64}
    D::Diagonal{Float64, Vector{Float64}}
    L_beta::Matrix{Float64}
    Lambda_hat::Vector{Float64}
    z::Vector{Float64}
    Lambda::Vector{Float64}
    L_R::Matrix{Float64}
    beta_tilde::Matrix{Float64}
    a_tilde::Vector{Float64}
    prob::Vector{Float64}
    gamma::BitVector
    resid::Vector{Float64}
    scale2::Vector{Float64}
    scale3::Vector{Float64}
    temp_mul1::Matrix{Float64}
    temp_mul2::Vector{Float64}
    temp_mul3::Matrix{Float64}
    temp_diag::Vector{Float64}
    cholesky_buffer::Matrix{Float64}
    sd_Y_outer::Matrix{Float64}
    temp_lambda_f::Vector{Float64}
    temp_sdf::Vector{Float64}
    bernoulli_dists::Vector{Bernoulli{Float64}}  # One for each factor
    beta_dists::Vector{Beta{Float64}}            # One for each fact
end

function MCMCTemps(intercept::Bool, p::Int, N::Int, k::Int, t::Int)
    k2 = intercept ? k+1 : k
    return MCMCTemps(
        zeros(Float64, p,p),            # Sigma
        zeros(Float64, p,p),            # Var_mu_half
        zeros(Float64,p),               # mu
        zeros(Float64,p),               # sd_Y
        zeros(Float64, p,p),            # corr_Y
        zeros(Float64, N,N),            # corrR
        ones(Float64, N,k2),            # beta
        zeros(Float64,N),               # a
        Diagonal(ones(k2)),             # D
        Matrix{Float64}(undef, k2, k2), # L_beta
        Vector{Float64}(undef, k2),     # Lambda_hat
        Vector{Float64}(undef, k2),     # z
        Vector{Float64}(undef, k2),     # Lambda
        Matrix{Float64}(undef, N, N),   # L_R
        Matrix{Float64}(undef, N, k2),  # beta_tilde
        Vector{Float64}(undef, N),      # a_tilde
        Vector{Float64}(undef, k),      # prob
        BitVector(undef, k),            # gamma
        Vector{Float64}(undef, N),      # resid
        [0.],                           # scale2
        [0.],                           # scale3
        Matrix{Float64}(undef, p, p),   # temp_mul1
        Vector{Float64}(undef, p),      # temp_mul2
        Matrix{Float64}(undef, p, p),   # temp_mul3
        Vector{Float64}(undef, k),      # temp_diag
        Matrix{Float64}(undef, p, p),   # cholesky_buffer
        Matrix{Float64}(undef, p, p),   # sd_Y_outer
        Vector{Float64}(undef, k),      # temp_lambda_f
        Vector{Float64}(undef, t),       # temp_sdf
        [Bernoulli(0.5) for _ in 1:k],  # Will update prob parameter each iteration
        [Beta(1.0, 1.0) for _ in 1:k]
    )
end

function compute_log_odds!(result::Vector{Float64}, 
    last_state::MCMCStates, 
    con::MCMCConstants, 
    Lambda::Vector{Float64})
    
    log_r_term = 0.5 * log(con.r)
    r_factor = 0.5 * (1/con.r - 1)
    
    @inbounds for i in eachindex(result)
        omega_ratio = last_state.omega[i] / (1 - last_state.omega[i])
        sigma_psi = last_state.sigma2 * con.psi[i]
        
        result[i] = log(omega_ratio) + log_r_term
        
        if con.intercept
            result[i] += r_factor * Lambda[i+1]^2 / sigma_psi
        else
            result[i] += r_factor * Lambda[i]^2 / sigma_psi
        end
    end
    return result
end

function get_sdf!(temp::MCMCTemps, con::MCMCConstants)
    # Use std function to match original
    @inbounds for i in 1:con.k
        temp.temp_diag[i] = std(view(con.f, :, i))
    end
    
    if con.intercept
        @inbounds for i in 1:con.k
            temp.temp_lambda_f[i] = temp.Lambda[i+1] / temp.temp_diag[i]
        end
    else
        @inbounds for i in 1:con.k
            temp.temp_lambda_f[i] = temp.Lambda[i] / temp.temp_diag[i]
        end
    end
    
    # Match original order of operations
    mul!(temp.temp_sdf, con.f, temp.temp_lambda_f)
    @inbounds for i in eachindex(temp.temp_sdf)
        temp.temp_sdf[i] = 1 - temp.temp_sdf[i]  # First subtract from 1
    end
    sdf_mean = sum(temp.temp_sdf) / length(temp.temp_sdf)
    
    @inbounds for i in eachindex(temp.temp_sdf)
        temp.temp_sdf[i] = 1 + temp.temp_sdf[i] - sdf_mean  # Match original addition and mean subtraction
    end
    
    return temp.temp_sdf
end

function mcmc_step!(con::MCMCConstants, last_state::MCMCStates, output::MCMCOutputs, i::Int, temp::MCMCTemps, k1::Int=con.k)
    mtwist = con.rngs

    
    # Draw new covariance matrix from inverse Wishart
    Random.seed!(mtwist, i)
    rand!(mtwist, con.iw_dist, temp.Sigma)
    
    # Calculate standardized quantities
    copyto!(temp.Var_mu_half, cholesky(Hermitian(temp.Sigma/con.t)).U)
    Random.seed!(mtwist, i)
    copyto!(temp.mu, con.mu_ols + transpose(temp.Var_mu_half) * randn(mtwist, con.p))
    copyto!(temp.sd_Y, sqrt.(diag(temp.Sigma)))
    
    # Compute correlation matrix and relevant submatrices
    copyto!(temp.corr_Y, temp.Sigma)
    temp.corr_Y ./= (temp.sd_Y * transpose(temp.sd_Y))
    
    # Use k1 instead of k for indexing the test assets
    copyto!(temp.corrR, temp.corr_Y[k1+1:con.p, k1+1:con.p])
    if con.intercept
        copyto!(view(temp.beta, :, 2:size(temp.beta,2)), temp.corr_Y[k1+1:con.p, 1:con.k])
    else
        copyto!(temp.beta, temp.corr_Y[k1+1:con.p, 1:con.k])
    end
    
    copyto!(temp.a, temp.mu[k1+1:con.p])
    temp.a ./= temp.sd_Y[k1+1:con.p]

    # Rest of the function remains exactly the same
    if con.intercept
        copyto!(temp.D, Diagonal(vcat([1 / 100000], 1 ./ (last_state.r_gamma .* con.psi))))
    else
        if con.k == 1
            copyto!(temp.D, Diagonal(fill(1/(last_state.r_gamma[1] * con.psi[1]), 1, 1)))
        else
            copyto!(temp.D, Diagonal(1 ./ (last_state.r_gamma .* con.psi)))
        end
    end

    if con.type == "OLS"
        mul!(temp.L_beta, transpose(temp.beta), temp.beta)
        temp.L_beta .+= temp.D
        cholesky!(temp.L_beta)
        transpose!(UpperTriangular(temp.L_beta))
        mul!(temp.Lambda, transpose(temp.beta), temp.a)
        ldiv!(temp.Lambda, LowerTriangular(temp.L_beta), temp.Lambda)
        ldiv!(temp.Lambda_hat, transpose(LowerTriangular(temp.L_beta)), temp.Lambda)
        Random.seed!(mtwist, i)
        randn!(mtwist, temp.z)
        ldiv!(temp.Lambda, transpose(LowerTriangular(temp.L_beta)), temp.z)
        mul!(temp.Lambda, sqrt(last_state.sigma2), temp.Lambda)
        temp.Lambda .+= temp.Lambda_hat
    else
        copyto!(temp.L_R, temp.corrR)
        cholesky!(temp.L_R)
        transpose!(UpperTriangular(temp.L_R))
        ldiv!(temp.beta_tilde, LowerTriangular(temp.L_R), temp.beta)
        ldiv!(temp.a_tilde, LowerTriangular(temp.L_R), temp.a)
        mul!(temp.L_beta, transpose(temp.beta_tilde), temp.beta_tilde)
        temp.L_beta .+= temp.D
        cholesky!(temp.L_beta)
        transpose!(UpperTriangular(temp.L_beta))
        mul!(temp.Lambda, transpose(temp.beta_tilde), temp.a_tilde)
        ldiv!(temp.Lambda, LowerTriangular(temp.L_beta), temp.Lambda)
        ldiv!(temp.Lambda_hat, transpose(LowerTriangular(temp.L_beta)), temp.Lambda)
        Random.seed!(mtwist, i)
        randn!(mtwist, temp.z)
        ldiv!(temp.Lambda, transpose(LowerTriangular(temp.L_beta)), temp.z)
        mul!(temp.Lambda, sqrt(last_state.sigma2), temp.Lambda)
        temp.Lambda .+= temp.Lambda_hat
    end

    compute_log_odds!(temp.prob, last_state, con, temp.Lambda)
    @. temp.prob = exp(temp.prob)
    @. temp.prob = min(temp.prob, 1000.0)
    @. temp.prob = temp.prob / (1 + temp.prob)
    
    Random.seed!(mtwist, i)
    @. temp.gamma = rand(mtwist, Bernoulli(temp.prob))
    @. last_state.r_gamma = ifelse(temp.gamma == 1, 1.0, con.r)
    output.gamma_path[i, :] = temp.gamma
    
    Random.seed!(mtwist, i)
    @. last_state.omega = rand(mtwist, Beta(con.aw + temp.gamma, con.bw + 1 - temp.gamma))

    if con.type == "OLS"
        mul!(temp.resid, temp.beta, temp.Lambda)
        @. temp.resid = temp.a - temp.resid
        mul!(temp.scale2, transpose(temp.resid), temp.resid)
        mul!(temp.scale3, transpose(temp.Lambda) * temp.D, temp.Lambda)
        @. temp.scale2 += temp.scale3
        @. temp.scale2 /= 2
        Random.seed!(mtwist, i)
        last_state.sigma2 = rand(mtwist, InverseGamma(con.dof2, first(temp.scale2)))
    else
        mul!(temp.resid, temp.beta, temp.Lambda)
        @. temp.resid = temp.a - temp.resid
        ldiv!(temp.resid, LowerTriangular(temp.L_R), temp.resid)
        mul!(temp.scale2, transpose(temp.resid), temp.resid)
        mul!(temp.scale3, transpose(temp.Lambda) * temp.D, temp.Lambda)
        @. temp.scale2 += temp.scale3
        @. temp.scale2 /= 2
        Random.seed!(mtwist, i)
        last_state.sigma2 = rand(mtwist, InverseGamma(con.dof2, first(temp.scale2)))
    end

    output.lambda_path[i, :] = temp.Lambda
    output.sdf_path[i, :] = get_sdf!(temp, con)
    
    return nothing
end


"""
    continuous_ss_sdf(f::Matrix{Float64}, R::Matrix{Float64}, sim_length::Int;
                    psi0::Float64=1.0, r::Float64=0.001,
                    aw::Float64=1.0, bw::Float64=1.0,
                    type::String="OLS", intercept::Bool=true)

SDF model selection using continuous spike-and-slab prior.

# Arguments
- `f`: Matrix of factors with dimension ``t \\times k``
- `R`: Matrix of test assets with dimension ``t \\times N``
- `sim_length`: Length of MCMCs
- `psi0`: Hyperparameter in prior distribution of risk prices
- `r`: Hyperparameter for spike component (``\\ll 1``)
- `aw,bw`: Beta prior parameters for factor inclusion probability
- `type`: "OLS" or "GLS"
- `intercept`: Include intercept if true

# Returns
Returns a ContinuousSSSDFOutput struct containing:
- gamma_path::Matrix{Float64}: Matrix of size sim_length × k containing posterior draws of factor inclusion indicators.
- lambda_path::Matrix{Float64}: Matrix of size sim_length × (k+1) if intercept=true, or sim_length × k if false. Contains posterior draws of risk prices.
- sdf_path::Matrix{Float64}: Matrix of size sim_length × t containing posterior draws of the SDF.
- bma_sdf::Vector{Float64}: Vector of length t containing the Bayesian Model Averaged SDF.
- Metadata fields accessible via dot notation:
 - n_factors::Int: Number of factors (k)
 - n_assets::Int: Number of test assets (N)
 - n_observations::Int: Number of time periods (t)
 - sim_length::Int: Number of MCMC iterations performed

# Notes
- Input matrices f and R must have the same number of rows (time periods)
- The method automatically handles both traded and non-traded factors
- Prior parameters aw, bw control beliefs about model sparsity (default values favor no sparsity)
- Parameter psi0 maps into prior beliefs about achievable Sharpe ratios
- The spike component r should be close to zero to effectively shrink irrelevant factors
- The resulting SDF is normalized to have mean 1

# References
Bryzgalova S, Huang J, Julliard C (2023). "Bayesian solutions for the factor zoo: We just ran two quadrillion models." Journal of Finance, 78(1), 487–557.

# Examples
```julia
# Basic usage with default settings
results = continuous_ss_sdf(f, R, 10_000)

# Use GLS with modified priors for more aggressive selection
results_gls = continuous_ss_sdf(f, R, 10_000;
                             type="GLS",
                             psi0=0.5,     # Tighter prior
                             aw=1.0,       
                             bw=9.0)       # Prior favoring sparsity

# Access results
inclusion_probs = mean(results.gamma_path, dims=1)  # Factor inclusion probabilities
risk_prices = mean(results.lambda_path, dims=1)     # Posterior mean risk prices
sdf = results.bma_sdf                              # Model averaged SDF
```
"""
function continuous_ss_sdf(f::Matrix{Float64}, R::Matrix{Float64}, sim_length::Int;
    psi0::Float64=1.0, r::Float64=0.001,
    aw::Float64=1.0, bw::Float64=1.0,
    type::String="OLS", intercept::Bool=true)
    
    # Initialize random number generators
    mtwist = MersenneTwister(1)
    
    # Get dimensions
    t, k = size(f)
    N = size(R, 2)
    p = k + N
    
    # Combine data and compute initial statistics
    Y = hcat(f, R)
    Sigma_ols = cov(Y)
    Corr_ols = cor(Y)
    sd_ols = vec(std(Y, dims=1))
    mu_ols = vec(mean(Y, dims=1))
    
    # Create initial beta matrix
    if intercept
        beta_ols = hcat(ones(N), Corr_ols[k+1:p, 1:k])
    else
        beta_ols = Corr_ols[k+1:p, 1:k]
    end
    
    # Initialize a_ols
    a_ols = mu_ols[k+1:p] ./ sd_ols[k+1:p]
    
    # Calculate initial Lambda
    L_init = cholesky(Hermitian(transpose(beta_ols)*beta_ols)).L
    Lambda_ols = L_init' \ (L_init \ (transpose(beta_ols) * a_ols))
    
    # Calculate psi
    rho = cor(Y)[k+1:p, 1:k]
    if intercept
        rho_mean = vec(mean(rho, dims=1))
        rho_demean = rho .- transpose(rho_mean)
    else
        rho_demean = rho
    end
    
    psi = if k == 1
        [psi0 * (transpose(rho_demean) * rho_demean)[1]]
    else
        psi0 * diag(transpose(rho_demean) * rho_demean)
    end
    
    # Pre-compute distributions and constants
    bernoulli_dist = [Bernoulli(0.5) for _ in 1:k]
    beta_dist_base = [Beta(aw + 1, bw + 1) for _ in 1:k]
    dof2 = intercept ? (N + k + 1) / 2 : (N + k) / 2
    
    # Initialize outputs
    output = MCMCOutputs(
        zeros(sim_length, k),
        zeros(sim_length, intercept ? k + 1 : k),
        zeros(sim_length, t)
    )
    
    # Initialize constants
    con = MCMCConstants(
        f, psi, r, aw, bw, type, intercept, mtwist,
        t, N, k, p, Y, InverseWishart(t - 1, t * Sigma_ols),
        mu_ols, dof2
    )
    
    # Initialize state
    Random.seed!(mtwist, 1)
    last_state = MCMCStates(
        ifelse.(rand(mtwist, Bernoulli(0.5), k) .== 1, 1.0, r),
        (transpose(a_ols - beta_ols * Lambda_ols) * (a_ols - beta_ols * Lambda_ols))[1] / N,
        fill(0.5, k)
    )
    
    # Initialize temps
    temp = MCMCTemps(con.intercept, con.p, con.N, con.k,t)
    
    # MCMC loop
    for i in 1:sim_length
        mcmc_step!(con, last_state, output, i, temp)
    end
    
    # Calculate BMA-SDF
    bma_sdf = vec(mean(output.sdf_path, dims=1))
    
    return ContinuousSSSDFOutput(
        output.gamma_path,
        output.lambda_path,
        output.sdf_path,
        bma_sdf;
        n_factors=k,
        n_assets=N,
        n_observations=t,
        sim_length=sim_length
    )
end
