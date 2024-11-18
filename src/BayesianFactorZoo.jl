module BayesianFactorZoo


using LinearAlgebra
using Distributions
using Statistics
using Random
using Combinatorics
using Printf

# Export main functions
export BayesianFM, # Equiv Output
    BayesianSDF, # Equiv Output
    continuous_ss_sdf, # Equiv Output
    continuous_ss_sdf_v2, # Equiv Output
    dirac_ss_sdf_pvalue,
    psi_to_priorSR, # Equiv Output
    SDF_gmm,
    TwoPassRegression, # Equiv Output
    BayesianFactorModel,
    BayesianFMOutput,
    BayesianSDFOutput,
    ContinuousSSSDFOutput,
    DiracSSSDFOutput,
    SDFGMMOutput,
    TwoPassRegressionOutput,
    summary_statistics,
    summary_table,
    JTestOutput,
    JTest

# Include Structs:
include("structs.jl")
# Include utility functions
include("utils.jl")

# Include main functionality
include("two_pass_regression.jl")
include("bayesian_fm.jl")
include("bayesian_sdf.jl")
include("continuous_ss_sdf.jl")
include("continuous_ss_sdf_v2.jl")
include("dirac_ss_sdf_pvalue.jl")
include("sdf_gmm.jl")

end # module