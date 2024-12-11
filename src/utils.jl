"""
Check input validity for Bayesian estimation functions
"""
function check_input(f::Matrix{Float64}, R::Matrix{Float64}, intercept::Bool, type::String, prior::String)
    t_f, k = size(f)    # number of factors k and time periods t_f
    t_asset, N = size(R) # number of test assets N and time periods t_asset

    # Check time periods match
    if t_f != t_asset
        throw(ArgumentError("Time periods of factors must equal time periods of assets"))
    end

    # Check dimensions for intercept case
    if intercept && k >= N
        throw(ArgumentError("Number of test assets must be larger than number of factors when including intercept"))
    end

    # Check dimensions for no intercept case
    if !intercept && k > N
        throw(ArgumentError("Number of test assets must be >= number of factors when excluding intercept"))
    end

    # Check type argument
    if !(type in ["OLS", "GLS"])
        throw(ArgumentError("type must be 'OLS' or 'GLS'"))
    end

    # Check prior argument
    if !(prior in ["Flat", "Spike-and-Slab", "Normal"])
        throw(ArgumentError("prior must be 'Flat', 'Spike-and-Slab', or 'Normal'"))
    end
end


"""
    check_input2(f::Matrix{Float64}, R::Matrix{Float64})

Check basic input validity for factor models.

# Arguments
- `f`: Matrix of factors
- `R`: Matrix of test assets

# Checks
1. Time periods match between factors and assets
2. Number of test assets > number of factors
"""
function check_input2(f::Matrix{Float64}, R::Matrix{Float64})
    t_f, k = size(f)    # number of factors k and time periods t_f
    t_asset, N = size(R) # number of test assets N and time periods t_asset

    # Check time periods match
    if t_f != t_asset
        throw(ArgumentError("Time periods of factors must equal time periods of assets"))
    end

    # Check dimensions requirement
    if k >= N
        throw(ArgumentError("Number of test assets must be larger (>) than number of factors"))
    end

    return nothing
end


"""
    psi_to_priorSR(R::Matrix{Float64}, f::Matrix{Float64}; 
                  psi0::Union{Nothing,Float64}=nothing,
                  priorSR::Union{Nothing,Float64}=nothing,
                  aw::Float64=1.0, bw::Float64=1.0)

Map between prior tightness parameter ``\\psi`` and prior Sharpe ratio.

# Arguments
- `R`: Matrix of test assets with dimension ``t \\times N``
- `f`: Matrix of factors with dimension ``t \\times k``
- `psi0`: Prior tightness parameter to convert to SR
- `priorSR`: Target SR to convert to psi0
- `aw,bw`: Beta prior parameters


# Returns
Returns a Float64 value either:
- The implied prior Sharpe ratio if psi0 provided
- The required psi0 value if priorSR provided

Note: Returns error message string if neither or both arguments are provided

# Notes
- Exactly one of psi0 or priorSR must be provided
- Input matrices R and f must have the same number of rows (time periods)
- The mapping helps choose priors based on economic intuition about achievable Sharpe ratios
- Default aw=bw=1 implies 50% prior probability of factor inclusion
- The relationship is monotonic: higher ψ implies higher prior Sharpe ratio
- Useful for calibrating priors in continuous_ss_sdf and continuous_ss_sdf_v2

# References
Bryzgalova S, Huang J, Julliard C (2023). "Bayesian solutions for the factor zoo: We just ran two quadrillion models." Journal of Finance, 78(1), 487–557.

# Examples
```julia
# Load test data
# Convert psi0 to implied prior Sharpe ratio
implied_sr = psi_to_priorSR(R, f; psi0=5.0)
println("Psi0 = 5.0 implies prior SR = \$implied_sr")

# Find psi0 needed for target Sharpe ratio
required_psi = psi_to_priorSR(R, f; priorSR=0.5)
println("For prior SR = 0.5, need psi0 = \$required_psi")

# Use custom Beta prior parameters
psi_sparse = psi_to_priorSR(R, f; 
                          priorSR=0.3,
                          aw=1.0, 
                          bw=9.0)  # Prior favoring sparsity

# Helper functions also available:
using BayesianFactorZoo: calculate_prior_SR, find_psi_for_target_SR

# Get prior SR for a given psi
sr = calculate_prior_SR(5.0, R, f)

# Get psi for a target SR
psi = find_psi_for_target_SR(0.5, R, f)
```

# See Also
- `continuous_ss_sdf`: Main function using this prior calibration
- `continuous_ss_sdf_v2`: Version for tradable factors using this calibration
"""
function psi_to_priorSR(R::Matrix{Float64}, f::Matrix{Float64};
    psi0::Union{Nothing,Float64}=nothing,
    priorSR::Union{Nothing,Float64}=nothing,
    aw::Float64=1.0, bw::Float64=1.0)

    # Check dimensions
    T_R, N = size(R)
    T_f, K = size(f)

    if T_R != T_f
        throw(DimensionMismatch("Number of time periods in R ($T_R) and f ($T_f) must match"))
    end

    if (isnothing(psi0) && isnothing(priorSR)) || (!isnothing(psi0) && !isnothing(priorSR))
        return "Please enter either psi0 or priorSR!"
    end

    # Calculate in-sample squared Sharpe ratio
    function SharpeRatio(R::Matrix{Float64})
        ER = mean(R, dims=1)'
        covR = cov(R)
        # Add small regularization if needed
        if !isposdef(covR)
            covR += 1e-6 * I(size(covR, 1))
        end
        return (ER'*inv(covR)*ER)[1]
    end

    SR_max = sqrt(SharpeRatio(R))
    corr_Rf = cor(R, f)  # This computes correlation properly for matrices

    # Cross-sectionally demean correlations
    corr_Rf_demean = corr_Rf .- mean(corr_Rf, dims=1)

    # Calculate eta parameter
    eta = (aw / (aw + bw)) * sum(diag(corr_Rf_demean' * corr_Rf_demean)) / N

    if isnothing(psi0) && !isnothing(priorSR)
        # Convert prior Sharpe ratio to psi0
        return priorSR^2 / ((SR_max^2 - priorSR^2) * eta)
    else
        # Convert psi0 to prior Sharpe ratio
        return sqrt((psi0 * eta / (1 + psi0 * eta))) * SR_max
    end
end

"""
    calculate_prior_SR(psi::Float64, R::Matrix{Float64}, f::Matrix{Float64}, 
                      aw::Float64=1.0, bw::Float64=1.0)

Helper function to calculate prior Sharpe ratio for a given psi value.
"""
function calculate_prior_SR(psi::Float64, R::Matrix{Float64}, f::Matrix{Float64},
    aw::Float64=1.0, bw::Float64=1.0)
    return psi_to_priorSR(R, f; psi0=psi, aw=aw, bw=bw)
end

"""
    find_psi_for_target_SR(target_SR::Float64, R::Matrix{Float64}, f::Matrix{Float64}, 
                          aw::Float64=1.0, bw::Float64=1.0)

Helper function to find psi value that gives a target Sharpe ratio.
"""
function find_psi_for_target_SR(target_SR::Float64, R::Matrix{Float64}, f::Matrix{Float64},
    aw::Float64=1.0, bw::Float64=1.0)
    return psi_to_priorSR(R, f; priorSR=target_SR, aw=aw, bw=bw)
end