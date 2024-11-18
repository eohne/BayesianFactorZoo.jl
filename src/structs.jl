"""
Core data structures for BayesianFactorZoo package
"""

# Abstract type for all model outputs
abstract type BayesianFactorModel end

"""
    BayesianFMOutput

Output type for Bayesian Fama-MacBeth regression.
Maintains same field names as original implementation for compatibility.
"""
struct BayesianFMOutput <: BayesianFactorModel
    # Original named tuple fields
    lambda_ols_path::Matrix{Float64}
    lambda_gls_path::Matrix{Float64}
    R2_ols_path::Vector{Float64}
    R2_gls_path::Vector{Float64}
    
    # Metadata
    metadata::Dict{Symbol,Any}
    
    # Inner constructor with validation
    function BayesianFMOutput(lambda_ols_path::Matrix{Float64},
                             lambda_gls_path::Matrix{Float64},
                             R2_ols_path::Vector{Float64},
                             R2_gls_path::Vector{Float64};
                             n_factors::Int=size(lambda_ols_path, 2)-1,
                             n_assets::Int=-1,
                             n_observations::Int=-1,
                             sim_length::Int=size(lambda_ols_path, 1))
        metadata = Dict{Symbol,Any}(
            :n_factors => n_factors,
            :n_assets => n_assets,
            :n_observations => n_observations,
            :sim_length => sim_length
        )
        new(lambda_ols_path, lambda_gls_path, R2_ols_path, R2_gls_path, metadata)
    end
end

"""
    BayesianSDFOutput

Output type for Bayesian SDF estimation.
"""
struct BayesianSDFOutput <: BayesianFactorModel
    # Original named tuple fields
    lambda_path::Matrix{Float64}
    R2_path::Vector{Float64}
    
    # Metadata
    metadata::Dict{Symbol,Any}
    
    function BayesianSDFOutput(lambda_path::Matrix{Float64},
                              R2_path::Vector{Float64};
                              n_factors::Int=size(lambda_path, 2)-1,
                              n_assets::Int=-1,
                              n_observations::Int=-1,
                              sim_length::Int=size(lambda_path, 1),
                              prior::String="Flat",
                              estimation_type::String="OLS")
        metadata = Dict{Symbol,Any}(
            :n_factors => n_factors,
            :n_assets => n_assets,
            :n_observations => n_observations,
            :sim_length => sim_length,
            :prior => prior,
            :estimation_type => estimation_type
        )
        new(lambda_path, R2_path, metadata)
    end
end

"""
    ContinuousSSSDFOutput

Output type for continuous spike-and-slab SDF estimation.
"""
struct ContinuousSSSDFOutput <: BayesianFactorModel
    # Original named tuple fields
    gamma_path::Matrix{Float64}
    lambda_path::Matrix{Float64}
    sdf_path::Matrix{Float64}
    bma_sdf::Vector{Float64}
    
    # Metadata
    metadata::Dict{Symbol,Any}
    
    function ContinuousSSSDFOutput(gamma_path::Matrix{Float64},
                                  lambda_path::Matrix{Float64},
                                  sdf_path::Matrix{Float64},
                                  bma_sdf::Vector{Float64};
                                  n_factors::Int=size(lambda_path, 2)-1,
                                  n_assets::Int=-1,
                                  n_observations::Int=length(bma_sdf),
                                  sim_length::Int=size(lambda_path, 1))
        metadata = Dict{Symbol,Any}(
            :n_factors => n_factors,
            :n_assets => n_assets,
            :n_observations => n_observations,
            :sim_length => sim_length
        )
        new(gamma_path, lambda_path, sdf_path, bma_sdf, metadata)
    end
end

"""
    DiracSSSDFOutput

Output type for Dirac spike-and-slab SDF estimation.
"""
struct DiracSSSDFOutput <: BayesianFactorModel
    # Core outputs - changed order to match fields used in constructor
    gamma_path::Matrix{Float64}    # Put first
    lambda_path::Matrix{Float64}   # Put second
    model_probs::Matrix{Float64}
    
    # Metadata
    metadata::Dict{Symbol,Any}
    
    function DiracSSSDFOutput(gamma_path::Matrix{Float64},
                             lambda_path::Matrix{Float64},
                             model_probs::Matrix{Float64};
                             n_factors::Int=size(lambda_path, 2)-1,
                             n_assets::Int=-1,
                             n_observations::Int=-1,
                             sim_length::Int=size(lambda_path, 1))
        metadata = Dict{Symbol,Any}(
            :n_factors => n_factors,
            :n_assets => n_assets,
            :n_observations => n_observations,
            :sim_length => sim_length
        )
        new(gamma_path, lambda_path, model_probs, metadata)
    end
end

"""
    SDFGMMOutput

Output type for SDF GMM estimation.
"""
struct SDFGMMOutput <: BayesianFactorModel
    # Original named tuple fields
    lambda_gmm::Vector{Float64}
    mu_f::Vector{Float64}
    Avar_hat::Matrix{Float64}
    R2_adj::Float64
    S_hat::Matrix{Float64}
    
    # Metadata
    metadata::Dict{Symbol,Any}
    
    function SDFGMMOutput(lambda_gmm::Vector{Float64},
                         mu_f::Vector{Float64},
                         Avar_hat::Matrix{Float64},
                         R2_adj::Float64,
                         S_hat::Matrix{Float64};
                         n_factors::Int,
                         n_assets::Int,
                         n_observations::Int)
        metadata = Dict{Symbol,Any}(
            :n_factors => n_factors,
            :n_assets => n_assets,
            :n_observations => n_observations
        )
        new(lambda_gmm, mu_f, Avar_hat, R2_adj, S_hat, metadata)
    end
end

"""
    TwoPassRegressionOutput

Output type for two-pass regression estimation.
"""
struct TwoPassRegressionOutput <: BayesianFactorModel
    # Core outputs
    lambda::Vector{Float64}
    lambda_gls::Vector{Float64}
    t_stat::Vector{Float64}
    t_stat_gls::Vector{Float64}
    R2_adj::Float64
    R2_adj_GLS::Float64
    alpha::Vector{Float64}
    t_alpha::Vector{Float64}
    beta::Matrix{Float64}
    cov_epsilon::Matrix{Float64}
    cov_lambda::Matrix{Float64}
    cov_lambda_gls::Matrix{Float64}
    R2_GLS::Float64
    cov_beta::Matrix{Float64}
    
    # Metadata
    metadata::Dict{Symbol,Any}
    
    function TwoPassRegressionOutput(lambda::Vector{Float64},
                                   lambda_gls::Vector{Float64},
                                   t_stat::Vector{Float64},
                                   t_stat_gls::Vector{Float64},
                                   R2_adj::Float64,
                                   R2_adj_GLS::Float64,
                                   alpha::Vector{Float64},
                                   t_alpha::Vector{Float64},
                                   beta::Matrix{Float64},
                                   cov_epsilon::Matrix{Float64},
                                   cov_lambda::Matrix{Float64},
                                   cov_lambda_gls::Matrix{Float64},
                                   R2_GLS::Float64,
                                   cov_beta::Matrix{Float64};
                                   n_factors::Int,
                                   n_assets::Int,
                                   n_observations::Int)
        metadata = Dict{Symbol,Any}(
            :n_factors => n_factors,
            :n_assets => n_assets,
            :n_observations => n_observations
        )
        new(lambda, lambda_gls, t_stat, t_stat_gls,
            R2_adj, R2_adj_GLS, alpha, t_alpha,
            beta, cov_epsilon, cov_lambda, cov_lambda_gls,
            R2_GLS, cov_beta, metadata)
    end
end

# Accessor methods

# Delete all the previous @eval Base.getproperty blocks and replace with these:

function Base.getproperty(x::BayesianFactorModel, s::Symbol)
    if s ∈ fieldnames(typeof(x))
        return getfield(x, s)
    else
        return getfield(x, :metadata)[s]
    end
end

function Base.propertynames(x::BayesianFactorModel)
    return (fieldnames(typeof(x))..., keys(getfield(x, :metadata))...)
end


# BayesianFMOutput
function Base.getproperty(x::BayesianFMOutput, s::Symbol)
    if s ∈ fieldnames(BayesianFMOutput)
        return getfield(x, s)
    else
        return getfield(x, :metadata)[s]
    end
end

function Base.propertynames(x::BayesianFMOutput)
    return (fieldnames(BayesianFMOutput)..., keys(getfield(x, :metadata))...)
end


# BayesianSDFOutput
function Base.getproperty(x::BayesianSDFOutput, s::Symbol)
    if s ∈ fieldnames(BayesianSDFOutput)
        return getfield(x, s)
    else
        return getfield(x, :metadata)[s]
    end
end

function Base.propertynames(x::BayesianSDFOutput)
    return (fieldnames(BayesianSDFOutput)..., keys(getfield(x, :metadata))...)
end


# ContinuousSSSDFOutput
function Base.getproperty(x::ContinuousSSSDFOutput, s::Symbol)
    if s ∈ fieldnames(ContinuousSSSDFOutput)
        return getfield(x, s)
    else
        return getfield(x, :metadata)[s]
    end
end

function Base.propertynames(x::ContinuousSSSDFOutput)
    return (fieldnames(ContinuousSSSDFOutput)..., keys(getfield(x, :metadata))...)
end


# DiracSSSDFOutput
function Base.getproperty(x::DiracSSSDFOutput, s::Symbol)
    if s ∈ fieldnames(DiracSSSDFOutput)
        return getfield(x, s)
    else
        return getfield(x, :metadata)[s]
    end
end

function Base.propertynames(x::DiracSSSDFOutput)
    return (fieldnames(DiracSSSDFOutput)..., keys(getfield(x, :metadata))...)
end

# SDFGMMOutput
function Base.getproperty(x::SDFGMMOutput, s::Symbol)
    if s ∈ fieldnames(SDFGMMOutput)
        return getfield(x, s)
    else
        return getfield(x, :metadata)[s]
    end
end

function Base.propertynames(x::SDFGMMOutput)
    return (fieldnames(SDFGMMOutput)..., keys(getfield(x, :metadata))...)
end


# TwoPassRegressionOutput
function Base.getproperty(x::TwoPassRegressionOutput, s::Symbol)
    if s ∈ fieldnames(TwoPassRegressionOutput)
        return getfield(x, s)
    else
        return getfield(x, :metadata)[s]
    end
end

function Base.propertynames(x::TwoPassRegressionOutput)
    return (fieldnames(TwoPassRegressionOutput)..., keys(getfield(x, :metadata))...)
end

# Summary statistics functions
function summary_statistics(model::BayesianFMOutput)
    Dict{Symbol,Any}(
        :lambda_ols_mean => vec(mean(model.lambda_ols_path, dims=1)),
        :lambda_ols_std => vec(std(model.lambda_ols_path, dims=1)),
        :lambda_gls_mean => vec(mean(model.lambda_gls_path, dims=1)),
        :lambda_gls_std => vec(std(model.lambda_gls_path, dims=1)),
        :R2_ols_mean => mean(model.R2_ols_path),
        :R2_gls_mean => mean(model.R2_gls_path)
    )
end

function summary_statistics(model::BayesianSDFOutput)
    Dict{Symbol,Any}(
        :lambda_mean => vec(mean(model.lambda_path, dims=1)),
        :lambda_std => vec(std(model.lambda_path, dims=1)),
        :R2_mean => mean(model.R2_path),
        :R2_std => std(model.R2_path)
    )
end

function summary_statistics(model::ContinuousSSSDFOutput)
    Dict{Symbol,Any}(
        :inclusion_probabilities => vec(mean(model.gamma_path, dims=1)),
        :lambda_mean => vec(mean(model.lambda_path, dims=1)),
        :lambda_std => vec(std(model.lambda_path, dims=1))
    )
end

function summary_statistics(model::DiracSSSDFOutput)
    Dict{Symbol,Any}(
        :inclusion_probabilities => vec(mean(model.gamma_path, dims=1)),
        :lambda_mean => vec(mean(model.lambda_path, dims=1)),
        :lambda_std => vec(std(model.lambda_path, dims=1))
    )
end


# function summary_statistics(model::SDFGMMOutput)
#     Dict{Symbol,Any}(
#         :lambda_se => sqrt.(diag(model.Avar_hat)),
#         :lambda_t => model.lambda_gmm ./ sqrt.(diag(model.Avar_hat))
#     )
# end



# Pretty printing for all types
function Base.show(io::IO, model::BayesianFactorModel)
    println(io, "$(typeof(model)):")
    println(io, "  Number of factors: $(model.n_factors)")
    println(io, "  Number of assets: $(model.n_assets)")
    println(io, "  Number of observations: $(model.n_observations)")
    if hasproperty(model, :sim_length)
        println(io, "  MCMC iterations: $(model.sim_length)")
    end
end

function Base.show(io::IO, ::MIME"text/plain", model::BayesianFMOutput)
    stats = summary_statistics(model)
    println(io, "Bayesian Fama-MacBeth Results:")
    println(io, "--------------------------------")
    println(io, "Model Information:")
    println(io, "  Number of factors: $(model.n_factors)")
    println(io, "  Number of assets: $(model.n_assets)")
    println(io, "  Time periods: $(model.n_observations)")
    println(io, "  MCMC iterations: $(model.sim_length)")
    println(io, "\nRisk Premia Estimates (OLS):")
    for i in 1:model.n_factors+1
        @printf(io, "  λ_%d: %.4f (±%.4f)\n", 
                i-1, stats[:lambda_ols_mean][i], stats[:lambda_ols_std][i])
    end
    println(io, "\nModel Fit:")
    @printf(io, "  Mean R² (OLS): %.4f\n", stats[:R2_ols_mean])
    @printf(io, "  Mean R² (GLS): %.4f\n", stats[:R2_gls_mean])
end

function Base.show(io::IO, ::MIME"text/plain", model::BayesianSDFOutput)
    stats = summary_statistics(model)
    println(io, "Bayesian SDF Results:")
    println(io, "--------------------------------")
    println(io, "Model Information:")
    println(io, "  Number of factors: $(model.n_factors)")
    println(io, "  Number of assets: $(model.n_assets)")
    println(io, "  Time periods: $(model.n_observations)")
    println(io, "  MCMC iterations: $(model.sim_length)")
    println(io, "  Prior type: $(model.prior)")
    println(io, "  Estimation type: $(model.estimation_type)")
    println(io, "\nRisk Price Estimates:")
    for i in 1:model.n_factors+1
        @printf(io, "  λ_%d: %.4f (±%.4f)\n", 
                i-1, stats[:lambda_mean][i], stats[:lambda_std][i])
    end
    println(io, "\nModel Fit:")
    @printf(io, "  Mean R²: %.4f (±%.4f)\n", stats[:R2_mean], stats[:R2_std])
end

function Base.show(io::IO, ::MIME"text/plain", model::ContinuousSSSDFOutput)
    stats = summary_statistics(model)
    println(io, "Continuous Spike-and-Slab SDF Results:")
    println(io, "--------------------------------")
    println(io, "Model Information:")
    println(io, "  Number of factors: $(model.n_factors)")
    println(io, "  Number of assets: $(model.n_assets)")
    println(io, "  Time periods: $(model.n_observations)")
    println(io, "  MCMC iterations: $(model.sim_length)")
    println(io, "\nFactor Inclusion Probabilities:")
    for i in 1:model.n_factors
        @printf(io, "  Factor %d: %.4f\n", i, stats[:inclusion_probabilities][i])
    end
    println(io, "\nRisk Price Estimates:")
    for i in 1:model.n_factors+1
        @printf(io, "  λ_%d: %.4f (±%.4f)\n", 
                i-1, stats[:lambda_mean][i], stats[:lambda_std][i])
    end
end

function Base.show(io::IO, ::MIME"text/plain", model::DiracSSSDFOutput)
    stats = summary_statistics(model)
    println(io, "Dirac Spike-and-Slab SDF Results:")
    println(io, "--------------------------------")
    println(io, "Model Information:")
    println(io, "  Number of factors: $(model.n_factors)")
    println(io, "  Number of assets: $(model.n_assets)")
    println(io, "  Time periods: $(model.n_observations)")
    println(io, "  MCMC iterations: $(model.sim_length)")
    println(io, "\nFactor Inclusion Probabilities:")
    for i in 1:model.n_factors
        @printf(io, "  Factor %d: %.4f\n", i, stats[:inclusion_probabilities][i])
    end
    println(io, "\nRisk Price Estimates:")
    for i in 1:model.n_factors+1
        @printf(io, "  λ_%d: %.4f (±%.4f)\n", 
                i-1, stats[:lambda_mean][i], stats[:lambda_std][i])
    end
end

function Base.show(io::IO, ::MIME"text/plain", model::SDFGMMOutput)
    # stats = summary_statistics(model)
    println(io, "SDF GMM Results:")
    println(io, "--------------------------------")
    println(io, "Model Information:")
    println(io, "  Number of factors: $(model.n_factors)")
    println(io, "  Number of assets: $(model.n_assets)")
    println(io, "  Time periods: $(model.n_observations)")
    println(io, "\nRisk Price Estimates:")
    # for i in 1:length(model.lambda_gmm)
    #     @printf(io, "  λ_%d: %.4f (SE: %.4f, t: %.2f)\n", 
    #             i-1, model.lambda_gmm[i], stats[:lambda_se][i], stats[:lambda_t][i])
    # end
    for i in 1:length(model.lambda_gmm)
        @printf(io, "  λ_%d: %.4f\n", 
                i-1, model.lambda_gmm[i])
    end
    println(io, "\nModel Fit:")
    @printf(io, "  Adjusted R²: %.4f\n", model.R2_adj)
end

function Base.show(io::IO, ::MIME"text/plain", model::TwoPassRegressionOutput)
    println(io, "Two-Pass Regression Results:")
    println(io, "--------------------------------")
    println(io, "Model Information:")
    println(io, "  Number of factors: $(model.n_factors)")
    println(io, "  Number of assets: $(model.n_assets)")
    println(io, "  Time periods: $(model.n_observations)")
    println(io, "\nOLS Risk Premia Estimates:")
    for i in 1:length(model.lambda)
        se = sqrt(model.cov_lambda[i,i])
        @printf(io, "  λ_%d: %.4f (SE: %.4f, t: %.2f)\n", 
                i-1, model.lambda[i], se, model.t_stat[i])
    end
    println(io, "\nGLS Risk Premia Estimates:")
    for i in 1:length(model.lambda_gls)
        se = sqrt(model.cov_lambda_gls[i,i])
        @printf(io, "  λ_%d: %.4f (SE: %.4f, t: %.2f)\n", 
                i-1, model.lambda_gls[i], se, model.t_stat_gls[i])
    end
    println(io, "\nModel Fit:")
    @printf(io, "  Adjusted R² (OLS): %.4f\n", model.R2_adj)
    @printf(io, "  Adjusted R² (GLS): %.4f\n", model.R2_adj_GLS)
end

# Convenience function to create summary tables
"""
    summary_table(model::BayesianFactorModel)

Create a summary table of model results.
"""
function summary_table(model::BayesianFactorModel)
    stats = summary_statistics(model)
    # Implementation depends on specific needs
    # Could return a DataFrame or formatted string
end