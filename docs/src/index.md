# BayesianFactorZoo.jl

BayesianFactorZoo.jl is a Julia port of the R package 'BayesianFactorZoo' ([CRAN link](https://cran.r-project.org/web/packages/BayesianFactorZoo/index.html)) implementing the econometric methods from the paper:

> Bryzgalova, S., Huang, J., & Julliard, C. (2023). Bayesian solutions for the factor zoo: We just ran two quadrillion models. Journal of Finance, 78(1), 487–557. [DOI: 10.1111/jofi.13197](https://doi.org/10.1111/jofi.13197)

> For a more detailed function documentations please see the documentation of the R package [Link to PDF](https://cran.r-project.org/web/packages/BayesianFactorZoo/BayesianFactorZoo.pdf)

## Overview

BayesianFactorZoo.jl provides a comprehensive framework for analyzing linear asset pricing models that is:
- Simple and robust
- Applicable to high-dimensional problems
- Capable of handling both tradable and non-tradable factors
- Valid under model misspecification
- Robust to weak factors

For a stand-alone model, the package delivers reliable price of risk estimates and detects weakly identified factors. For competing factors and models, it provides automatic model selection or Bayesian model averaging when no clear winner exists.

## Installation

For now this is not in the general registry (I will register it at some point). In the meantime you can install it directly from my repository.
```julia
using Pkg
Pkg.add("http://github.com/eohne/BayesianFactorZoo.jl")
```

## Quick Start

```julia
using BayesianFactorZoo

# Example with simulated data
t, k, N = 600, 3, 25  # time periods, factors, assets
f = randn(t, k)       # factor returns
R = randn(t, N)       # asset returns

# Perform Bayesian Fama-MacBeth regression
results_fm = BayesianFM(f, R, 10_000)

# Estimate SDF with normal prior
results_sdf = BayesianSDF(f, R; prior="Normal")

# Model selection with continuous spike-and-slab
results_ss = continuous_ss_sdf(f, R, 10_000)
```

## Features

- Bayesian Fama-MacBeth regression (`BayesianFM`)
- Bayesian SDF estimation (`BayesianSDF`)
- Model selection via spike-and-slab priors (`continuous_ss_sdf`, `continuous_ss_sdf_v2`)
- Hypothesis testing (`dirac_ss_sdf_pvalue`)
- GMM estimation (`SDF_gmm`)
- Classical two-pass regression (`TwoPassRegression`)

## Citation

If you use this package, please cite:

```bibtex
@article{bryzgalova2023bayesian,
  title={Bayesian solutions for the factor zoo: We just ran two quadrillion models},
  author={Bryzgalova, Svetlana and Huang, Jiantao and Julliard, Christian},
  journal={The Journal of Finance},
  volume={78},
  number={1},
  pages={487--557},
  year={2023}
}
```