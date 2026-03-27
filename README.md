# BayesianFactorZoo.jl

[![][docs-latest-img]][docs-latest-url]
[![CI](https://github.com/eohne/BayesianFactorZoo.jl/actions/workflows/CI.yml/badge.svg)]

BayesianFactorZoo.jl is a Julia port of the R package `BayesianFactorZoo` ([CRAN link](https://cran.r-project.org/web/packages/BayesianFactorZoo/index.html)) implementing the econometric methods from the paper:

> Bryzgalova, S., Huang, J., & Julliard, C. (2023). Bayesian solutions for the factor zoo: We just ran two quadrillion models. Journal of Finance, 78(1), 487-557. [DOI: 10.1111/jofi.13197](https://doi.org/10.1111/jofi.13197)

If you use this package, please cite the original paper.

For detailed documentation of the original R implementation, see the [R package documentation (PDF)](https://cran.r-project.org/web/packages/BayesianFactorZoo/BayesianFactorZoo.pdf).
Note that function signatures and exposed functions are equivalent to the R version.

## Installation

The package is registered in the [`General`](https://github.com/JuliaRegistries/General) registry and so can be installed at the REPL with `] add BayesianFactorZoo` or by running:
```julia 
    using Pkg 
    Pkg.add("BayesianFactorZoo")
```

Alternatively you can install the latest dev version directly from this repository.

```julia
using Pkg
Pkg.add(url="http://github.com/eohne/BayesianFactorZoo.jl")
```

> **For some examples see the "Tutorial" in the documentation**
> [Link to Tutorial](https://eohne.github.io/BayesianFactorZoo.jl/dev/tutorial/)

## Main Functions

### BayesianFM

```julia
BayesianFM(f::Matrix{Float64}, R::Matrix{Float64}, sim_length::Int; seed::Union{Nothing,Integer}=nothing)
```

Performs Bayesian Fama-MacBeth regression.

Parameters:
- `f`: t x k matrix of factors (t periods, k factors)
- `R`: t x N matrix of test asset returns
- `sim_length`: Length of MCMC chain
- `seed`: Random seed for reproducibility. `nothing` (default) uses fresh non-reproducible randomness; an integer gives reproducible results

Returns:
- `lambda_ols_path`: MCMC draws of OLS risk premia
- `lambda_gls_path`: MCMC draws of GLS risk premia
- `R2_ols_path`: MCMC draws of OLS R^2
- `R2_gls_path`: MCMC draws of GLS R^2

### BayesianSDF

```julia
BayesianSDF(f::Matrix{Float64}, R::Matrix{Float64}, sim_length::Int; kwargs...)
```

Performs Bayesian estimation of linear stochastic discount factor (SDF). Supports both flat and normal priors for risk prices.

Parameters:
- `f`: t x k matrix of factors
- `R`: t x N matrix of test asset returns
- `sim_length`: Length of MCMC chain
- `intercept`: Include intercept (default: true)
- `type`: `"OLS"` or `"GLS"` (default: `"OLS"`)
- `prior`: `"Flat"` or `"Normal"` (default: `"Flat"`)
- `psi0`: Prior hyperparameter (default: 5.0)
- `d`: Prior hyperparameter (default: 0.5)
- `seed`: Random seed for reproducibility. `nothing` (default) uses fresh non-reproducible randomness; an integer gives reproducible results

Returns:
- `lambda_path`: MCMC draws of risk prices
- `R2_path`: MCMC draws of R^2

### Model Selection and Testing

#### continuous_ss_sdf

```julia
continuous_ss_sdf(f::Matrix{Float64}, R::Matrix{Float64}, sim_length::Int; kwargs...)
```

Performs SDF model selection using continuous spike-and-slab prior.

Parameters:
- `f`: t x k matrix of factors
- `R`: t x N matrix of test asset returns
- `sim_length`: Length of MCMC chain
- `psi0`: Prior scale (default: 1.0)
- `r`: Spike component (default: 0.001)
- `aw`, `bw`: Beta prior parameters (default: 1.0)
- `type`: `"OLS"` or `"GLS"` (default: `"OLS"`)
- `intercept`: Include intercept (default: true)
- `seed`: Random seed for reproducibility. `nothing` (default) uses fresh non-reproducible randomness; an integer gives a reproducible chain

Returns:
- `gamma_path`: MCMC draws of model inclusion indicators
- `lambda_path`: MCMC draws of risk prices
- `sdf_path`: MCMC draws of SDF values
- `bma_sdf`: Bayesian Model Averaged SDF

#### continuous_ss_sdf_v2

```julia
continuous_ss_sdf_v2(f1::Matrix{Float64}, f2::Matrix{Float64}, R::Matrix{Float64}, sim_length::Int; kwargs...)
```

Performs SDF model selection using continuous spike-and-slab prior while treating tradable factors as test assets.

Parameters:
- `f1`: t x k1 matrix of nontradable factors
- `f2`: t x k2 matrix of tradable factors
- `R`: t x N matrix of test asset returns, excluding `f2`
- `sim_length`: Length of MCMC chain
- `psi0`: Prior scale (default: 1.0)
- `r`: Spike component (default: 0.001)
- `aw`, `bw`: Beta prior parameters (default: 1.0)
- `type`: `"OLS"` or `"GLS"` (default: `"OLS"`)
- `intercept`: Include intercept (default: true)
- `seed`: Random seed for reproducibility. `nothing` (default) uses fresh non-reproducible randomness; an integer gives a reproducible chain

Returns:
- `gamma_path`: MCMC draws of model inclusion indicators
- `lambda_path`: MCMC draws of risk prices
- `sdf_path`: MCMC draws of SDF values
- `bma_sdf`: Bayesian Model Averaged SDF

#### dirac_ss_sdf_pvalue

```julia
dirac_ss_sdf_pvalue(f::Matrix{Float64}, R::Matrix{Float64}, sim_length::Int, lambda0::Vector{Float64}; kwargs...)
```

Parameters:
- `f`: t x k matrix of factors
- `R`: t x N matrix of test asset returns
- `sim_length`: Length of MCMC chain
- `lambda0`: Vector of risk prices under null hypothesis
- `psi0`: Prior scale (default: 1.0)
- `max_k`: Maximum model size (optional)
- `seed`: Random seed for reproducibility. `nothing` (default) uses fresh non-reproducible randomness; an integer gives reproducible results

Returns:
- `gamma_path`: MCMC draws of inclusion indicators
- `lambda_path`: MCMC draws of risk prices
- `model_probs`: Posterior model probabilities

## Example Usage

```julia
using BayesianFactorZoo

# Generate example data
t, k, N = 100, 2, 25
f = randn(t, k)    # Factor returns (100 periods x 2 factors)
R = randn(t, N)    # Test asset returns (100 periods x 25 assets)

# Bayesian Fama-MacBeth regression
bfm_result = BayesianFM(f, R, 10000; seed=1234)
println("Mean OLS risk premia: ", mean(bfm_result.lambda_ols_path, dims=1))
println("Mean OLS R^2: ", mean(bfm_result.R2_ols_path))

# Bayesian SDF estimation with normal prior
bsdf_result = BayesianSDF(
    f,
    R,
    10000;
    intercept=true,
    type="OLS",
    prior="Normal",
    psi0=5.0,
    seed=1234,
)

# Model selection with continuous spike-and-slab prior
# Set prior Sharpe ratio through psi0
psi_value = psi_to_priorSR(R, f, priorSR=0.1)
css_result = continuous_ss_sdf(
    f,
    R,
    10000;
    psi0=psi_value,
    r=0.001,
    seed=1234,
)

# Check factor inclusion probabilities
inclusion_probs = mean(css_result.gamma_path, dims=1)
println("Factor inclusion probabilities: ", inclusion_probs)
```

## License

This package is licensed under GPL-3, as required due to being a port of the GPL-3 licensed R package BayesianFactorZoo.

## Related Links

- [Original Paper](https://doi.org/10.1111/jofi.13197)
- [R Package on CRAN](https://cran.r-project.org/web/packages/BayesianFactorZoo/index.html)

## Speed

The length of monte-carlo simulations was set to 100,000 and the example data of the R package was used. I ran each function 10 times and report the median values.
Timings are from a laptop with an Intel Ultra 9 185H using 18 threads. Multi-threading of the Julia version is indicated by `*`.

| Method | Julia Time (s) | R Time (s) | Speed Improvement |
|--------|---------------|------------|-------------------|
| SDF GMM | 0.0008 | 0.0050 | 6.4x |
| Bayesian FM* | 2.30 | 41.69 | 18.1x |
| Bayesian SDF* | 2.69 | 29.54 | 11.0x |
| Continuous SS SDF | 8.13 | 42.82 | 5.3x |
| Continuous SS SDF v2 | 10.51 | 44.17 | 4.2x |
| Dirac SS SDF P-value* | 2.09 | 41.64 | 19.9x |

[docs-latest-img]: https://img.shields.io/badge/docs-latest-blue.svg
[docs-latest-url]: https://eohne.github.io/BayesianFactorZoo.jl/dev/
