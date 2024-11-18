# BayesianFactorZoo.jl

BayesianFactorZoo.jl is a Julia port of the R package 'BayesianFactorZoo' ([CRAN link](https://cran.r-project.org/web/packages/BayesianFactorZoo/index.html)) implementing the econometric methods from the paper:

> Bryzgalova, S., Huang, J., & Julliard, C. (2023). Bayesian solutions for the factor zoo: We just ran two quadrillion models. Journal of Finance, 78(1), 487–557. [DOI: 10.1111/jofi.13197](https://doi.org/10.1111/jofi.13197)

If you use this package, please cite the original paper.

For detailed documentation of the original R implementation, see the [R package documentation (PDF)](https://cran.r-project.org/web/packages/BayesianFactorZoo/BayesianFactorZoo.pdf).
Note that function signatures and exposed functions are equivalent to the R version.

## Installation
For now this is not in the general registry (I will register it at some point). In the meantime you can install it directly from my repository.
```julia
using Pkg
Pkg.add("url-to-repo")
```

## Main Functions

### `BayesianFM(f::Matrix{Float64}, R::Matrix{Float64}, sim_length::Int)`

Performs Bayesian Fama-MacBeth regression.

Parameters:
- `f`: t × k matrix of factors (t periods, k factors)
- `R`: t × N matrix of test asset returns
- `sim_length`: Length of MCMC chain

Returns:
- `lambda_ols_path`: MCMC draws of OLS risk premia
- `lambda_gls_path`: MCMC draws of GLS risk premia
- `R2_ols_path`: MCMC draws of OLS R²
- `R2_gls_path`: MCMC draws of GLS R²

### `BayesianSDF(f::Matrix{Float64}, R::Matrix{Float64}, sim_length::Int; kwargs...)`

Performs Bayesian estimation of linear stochastic discount factor (SDF). Supports both flat and normal priors for risk prices.

Parameters:
- `f`: t × k matrix of factors
- `R`: t × N matrix of test asset returns
- `sim_length`: Length of MCMC chain
- `intercept`: Include intercept (default: true)
- `type`: "OLS" or "GLS" (default: "OLS")
- `prior`: "Flat" or "Normal" (default: "Flat")
- `psi0`: Prior hyperparameter (default: 5.0)
- `d`: Prior hyperparameter (default: 0.5)

Returns:
- `lambda_path`: MCMC draws of risk prices
- `R2_path`: MCMC draws of R²

### Model Selection and Testing

#### `continuous_ss_sdf(f::Matrix{Float64}, R::Matrix{Float64}, sim_length::Int; kwargs...)`

Performs SDF model selection using continuous spike-and-slab prior.

Parameters:
- `f`: t × k matrix of factors
- `R`: t × N matrix of test asset returns
- `sim_length`: Length of MCMC chain
- `psi0`: Prior scale (default: 1.0)
- `r`: Spike component (default: 0.001)
- `aw`, `bw`: Beta prior parameters (default: 1.0)
- `type`: "OLS" or "GLS" (default: "OLS")
- `intercept`: Include intercept (default: true)

Returns:
- `gamma_path`: MCMC draws of model inclusion indicators
- `lambda_path`: MCMC draws of risk prices
- `sdf_path`: MCMC draws of SDF values
- `bma_sdf`: Bayesian Model Averaged SDF

#### `dirac_ss_sdf_pvalue(f::Matrix{Float64}, R::Matrix{Float64}, sim_length::Int, lambda0::Vector{Float64})`

Parameters:
- `f`: t × k matrix of factors
- `R`: t × N matrix of test asset returns
- `sim_length`: Length of MCMC chain
- `lambda0`: Vector of risk prices under null hypothesis
- `psi0`: Prior scale (default: 1.0)
- `max_k`: Maximum model size (optional)

Returns:
- `gamma_path`: MCMC draws of inclusion indicators
- `lambda_path`: MCMC draws of risk prices
- `model_probs`: Posterior model probabilities

## Example Usage

```julia
using BayesianFactorZoo

# Generate example data
t, k, N = 100, 2, 25
f = randn(t, k)    # Factor returns (100 periods × 2 factors)
R = randn(t, N)    # Test asset returns (100 periods × 25 assets)

# Bayesian Fama-MacBeth regression
bfm_result = BayesianFM(f, R, 10000)
println("Mean OLS risk premia: ", mean(bfm_result.lambda_ols_path, dims=1))
println("Mean OLS R²: ", mean(bfm_result.R2_ols_path))

# Bayesian SDF estimation with normal prior
bsdf_result = BayesianSDF(f, R, 10000, 
                         intercept=true, 
                         type="OLS",
                         prior="Normal",
                         psi0=5.0)

# Model selection with continuous spike-and-slab prior
# Set prior Sharpe ratio through psi0
psi_value = psi_to_priorSR(R, f, priorSR=0.1)
css_result = continuous_ss_sdf(f, R, 10000, 
                             psi0=psi_value, 
                             r=0.001)

# Check factor inclusion probabilities
inclusion_probs = mean(css_result.gamma_path, dims=1)
println("Factor inclusion probabilities: ", inclusion_probs)
```

## License

This package is licensed under GPL-3, as required due to being a port of the GPL-3 licensed R package BayesianFactorZoo.

## Related Links

- [Original Paper](https://doi.org/10.1111/jofi.13197)
- [R Package on CRAN](https://cran.r-project.org/web/packages/BayesianFactorZoo/index.html)