"""
    continuous_ss_sdf_v2(f1::Matrix{Float64}, f2::Matrix{Float64}, R::Matrix{Float64},
                       sim_length::Int; psi0::Float64=1.0, r::Float64=0.001,
                       aw::Float64=1.0, bw::Float64=1.0,
                       type::String="OLS", intercept::Bool=true)

SDF model selection with continuous spike-and-slab prior, treating tradable factors as test assets.

# Arguments
- `f1`: Matrix of nontradable factors with dimension ``t \\times k_1``
- `f2`: Matrix of tradable factors with dimension ``t \\times k_2``
- `R`: Matrix of test assets with dimension ``t \\times N`` (should NOT contain f2)
- `sim_length`: Length of MCMCs
- `psi0,r,aw,bw,type,intercept`: Same as continuous_ss_sdf

# Details
Same prior structure and posterior distributions as continuous_ss_sdf, but:
1. Treats tradable factors f2 as test assets
2. Total dimension of test assets becomes ``N + k_2``
3. Factor loadings computed on combined test asset set


# Returns
Returns a ContinuousSSSDFOutput struct containing:
- `gamma_path::Matrix{Float64}`: Matrix of size sim_length × k containing posterior draws of factor inclusion indicators, where ``k = k_1 + k_2`` (total number of factors).
- `lambda_path::Matrix{Float64}`: Matrix of size sim_length × (k+1) if intercept=true, or sim_length × k if false. Contains posterior draws of risk prices.
- `sdf_path::Matrix{Float64}`: Matrix of size sim_length × t containing posterior draws of the SDF.
- `bma_sdf::Vector{Float64}`: Vector of length t containing the Bayesian Model Averaged SDF.
- Metadata fields accessible via dot notation:
 - `n_factors::Int`: Number of factors (``k_1 + k_2``)
 - `n_assets::Int`: Number of test assets (N)
 - `n_observations::Int`: Number of time periods (t)
 - `sim_length::Int`: Number of MCMC iterations performed

# Notes
- Input matrices f1, f2, and R must have the same number of rows (time periods)
- Test assets R should not include the tradable factors f2
- The factor selection combines both sparsity and density aspects through Bayesian Model Averaging
- Prior parameters aw, bw control beliefs about model sparsity
- Parameter psi0 maps into prior beliefs about achievable Sharpe ratios
- The spike component r should be close to zero to effectively shrink irrelevant factors

# References
Bryzgalova S, Huang J, Julliard C (2023). "Bayesian solutions for the factor zoo: We just ran two quadrillion models." Journal of Finance, 78(1), 487–557.

# Examples
```julia
# Basic usage with default settings
results = continuous_ss_sdf_v2(f1, f2, R, 10_000)

# Use GLS with custom priors
results_gls = continuous_ss_sdf_v2(f1, f2, R, 10_000;
                                type="GLS",
                                psi0=2.0,
                                aw=2.0, 
                                bw=2.0)

# Access results
inclusion_probs = mean(results.gamma_path, dims=1)  # Factor inclusion probabilities
risk_prices = mean(results.lambda_path, dims=1)     # Risk price estimates
avg_sdf = results.bma_sdf                          # Model averaged SDF
```
"""
function continuous_ss_sdf_v2(f1::Matrix{Float64}, f2::Matrix{Float64}, R::Matrix{Float64}, sim_length::Int;
    psi0::Float64=1.0, r::Float64=0.001,
    aw::Float64=1.0, bw::Float64=1.0,
    type::String="OLS", intercept::Bool=true)
    
    # Initialize random number generators
    mtwist = MersenneTwister(1)
    
    # Get dimensions
    t = size(f1, 1)
    k1 = size(f1, 2)
    k2 = size(f2, 2)
    k = k1 + k2
    N = size(R, 2) + k2  # Include tradable factors as test assets
    p = k1 + N
    
    # Calculate degrees of freedom
    dof2 = intercept ? (N + k + 1) / 2 : (N + k) / 2
    
    # Combine factors and create full data matrix
    f = hcat(f1, f2)
    Y = hcat(f, R)       # Now Y will have f1, f2, R
    
    # Compute initial statistics
    Sigma_ols = cov(Y)
    Corr_ols = cor(Y)
    sd_ols = vec(std(Y, dims=1))
    mu_ols = vec(mean(Y, dims=1))
    
    # Create initial beta matrix based on correlations
    if intercept
        beta_ols = hcat(ones(N), Corr_ols[k1+1:p, 1:k])
    else
        beta_ols = Corr_ols[k1+1:p, 1:k]
    end
    
    # Initialize a_ols for first sigma2 calculation
    a_ols = mu_ols[k1+1:p] ./ sd_ols[k1+1:p]
    
    # Calculate initial Lambda and sigma2 using Cholesky
    L_init = cholesky(Hermitian(transpose(beta_ols)*beta_ols)).L
    Lambda_ols = L_init' \ (L_init \ (transpose(beta_ols) * a_ols))
    
    # Calculate psi for prior distribution
    rho = cor(Y)[k1+1:p, 1:k]
    if intercept
        rho_mean = vec(mean(rho, dims=1))
        rho_demean = rho .- transpose(rho_mean)
    else
        rho_demean = rho
    end
    
    # Set psi based on correlation structure
    if k == 1
        psi = psi0 * (transpose(rho_demean) * rho_demean)[1]
    else
        psi = psi0 * diag(transpose(rho_demean) * rho_demean)
    end
    
    # Setup inverse Wishart distribution
    iw_dist = InverseWishart(t - 1, t * Sigma_ols)
    
    # Initialize Output
    output = MCMCOutputs(
        zeros(sim_length, k),
        zeros(sim_length, intercept ? k + 1 : k),
        zeros(sim_length, t)
    )
    
    # Initialize Constants with dof2 as the last parameter
    con = MCMCConstants(f, psi, r, aw, bw, type, intercept, mtwist, t, N, k, p, Y, iw_dist, mu_ols, dof2)
    
    # Initialize State Variables
    Random.seed!(mtwist, 1)
    last_state = MCMCStates(
        ifelse.(rand(mtwist, Bernoulli(0.5), k) .== 1, 1.0, r),
        (transpose(a_ols - beta_ols * Lambda_ols) * (a_ols - beta_ols * Lambda_ols))[1] / N,
        fill(0.5, k)
    )
    
    # Initialize temporary storage
    temp = MCMCTemps(con.intercept, con.p, N, con.k,t)
    
    # MCMC loop
    for i in 1:sim_length
        mcmc_step!(con, last_state, output, i, temp, k1)
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