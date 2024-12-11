# Tutorial

## Basic Usage

This tutorial demonstrates how to use BayesianFactorZoo.jl for estimating and selecting asset pricing models.

### Data Preparation

First, let's prepare some example data:

```julia
using BayesianFactorZoo
using Random
Random.seed!(1234)

# Generate example data
t, k, N = 600, 3, 25  # time periods, factors, assets
f = randn(t, k)       # factor returns
R = randn(t, N)       # asset returns
```

### Bayesian Fama-MacBeth Regression

```julia
# Run Bayesian FM with 10,000 iterations
results_fm = BayesianFM(f, R, 10_000)

# Analyze results
mean_risk_premia = mean(results_fm.lambda_ols_path, dims=1)
mean_r2 = mean(results_fm.R2_ols_path)
```

### Bayesian SDF Estimation

```julia
# Estimate SDF with normal prior
results_sdf = BayesianSDF(f, R; 
                         prior="Normal",
                         psi0=5.0)

# Analyze results
mean_risk_prices = mean(results_sdf.lambda_path, dims=1)
mean_r2 = mean(results_sdf.R2_path)
```

### Model Selection

```julia
# Run continuous spike-and-slab selection
results_ss = continuous_ss_sdf(f, R, 10_000;
                             psi0=1.0,
                             aw=1.0,
                             bw=1.0)

# Analyze factor inclusion probabilities
inclusion_probs = mean(results_ss.gamma_path, dims=1)

# Get model averaged SDF
bma_sdf = results_ss.bma_sdf
```

## Advanced Topics

### Prior Calibration

The package provides tools for setting economically meaningful priors:

```julia
# Convert between psi and Sharpe ratios
implied_sr = psi_to_priorSR(R, f; psi0=5.0)
required_psi = psi_to_priorSR(R, f; priorSR=0.5)
```

### Hypothesis Testing

Test specific hypotheses about risk prices:

```julia
# Test if risk prices are zero
lambda0 = zeros(k)
results_test = dirac_ss_sdf_pvalue(f, R, 10_000, lambda0)

# Analyze posterior probabilities
p_values = 1 .- mean(results_test.gamma_path, dims=1)
```

### Handling Tradable Factors

When some factors are tradable, use the v2 version:

```julia
k1, k2 = 2, 1  # 2 non-tradable, 1 tradable factor
f1 = f[:, 1:k1]
f2 = f[:, k1+1:end]

results_v2 = continuous_ss_sdf_v2(f1, f2, R, 10_000)
```