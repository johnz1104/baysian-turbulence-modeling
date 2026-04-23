# Bayesian Turbulence Modeling

## Purpose

RANS turbulence models underpin the vast majority of production CFD, yet their closure coefficients are empirical constants tuned decades ago for generic flow classes. When applied to a specific configuration — a particular geometry, Reynolds number regime, or separation-dominated flow — these defaults can introduce systematic prediction error with no indication of how wrong they might be. This project aims to replace that single-point-estimate guesswork with a principled statistical framework: given experimental or DNS observations of a flow, it learns a full posterior distribution over the SST k-omega closure coefficients, producing calibrated values together with rigorous uncertainty bounds. The result is not just a better set of constants, but a quantification of how much the data actually constrains each parameter.

## Why Bayesian Calibration

Classical turbulence model tuning treats coefficient selection as an optimization problem: find the single parameter vector that minimizes some error metric against reference data. This approach has three fundamental limitations:

1. **No uncertainty quantification.** A point estimate gives no indication of how sensitive predictions are to the chosen values, or how well the data actually constrains each coefficient. Two very different parameter sets may produce nearly identical fits to the calibration data but diverge on unseen flows.

2. **Overfitting to calibration data.** Optimization finds the single best fit, which may exploit noise in the reference measurements or compensating errors between parameters. Without a regularizing prior, nothing prevents the optimizer from pushing coefficients to unphysical values if doing so reduces the objective.

3. **No propagation of parametric uncertainty.** Downstream decisions (design margins, safety factors) require knowing how uncertain the CFD prediction is. A point estimate cannot provide this — only a distribution over parameters, propagated through the model, gives a predictive distribution with meaningful error bars.

Bayesian inference addresses all three. Bayes' theorem combines prior knowledge of physically reasonable coefficient ranges with the likelihood of observing the experimental data given a particular parameter set:

```
p(theta | data) ∝ p(data | theta) * p(theta)
```

The prior `p(theta)` encodes what is known before seeing data (Menter's defaults, positivity constraints, stability limits). The likelihood `p(data | theta)` measures how well the CFD prediction at parameter vector `theta` matches the observations. The posterior `p(theta | data)` is the full answer: a joint distribution over all closure coefficients that reflects both prior knowledge and experimental evidence. Marginals of this distribution give calibrated means, credible intervals, and inter-parameter correlations. Parameters that the data cannot constrain remain close to the prior; parameters the data is informative about shift and tighten.

## Pipeline End Goal & Overview

Direct MCMC over the posterior is infeasible because each likelihood evaluation requires a full CFD solve (minutes per evaluation, millions of evaluations needed). The pipeline uses **surrogate-accelerated MCMC**: a Gaussian process emulator trained on a modest ensemble of CFD runs replaces the expensive solver during sampling.

### Stage 1 — Prior Specification

A truncated normal distribution centered on Menter's (1994) SST defaults with 15% relative standard deviation. The truncation bounds enforce positivity and known stability constraints, assigning zero probability to parameter regions that would cause the solver to diverge or violate realizability.

```
log p(theta) = -0.5 * sum_i ((theta_i - mu_i) / sigma_i)^2    if theta in bounds
             = -inf                                             otherwise
```

### Stage 2 — Ensemble Generation (Latin Hypercube Sampling + Forward Model)

Latin Hypercube Sampling generates ~200 parameter vectors with uniform marginal coverage across the prior bounds — far more space-filling than random sampling for the same budget. Each vector is passed to the C++ forward model, which:

1. Unpacks the SST coefficients
2. Runs the SIMPLE pressure-velocity coupling solver (AMG for pressure, BiCGSTAB for momentum and turbulence equations)
3. Extracts observables via the observation operator (drag, skin friction, velocity profiles, pressure taps, separation point)
4. Evaluates a Gaussian log-likelihood against the reference data

Failed or diverged evaluations are filtered out; the surviving (theta, log-likelihood) pairs form the training set for the surrogate.

### Stage 3 — Gaussian Process Surrogate

A GP with an ARD (Automatic Relevance Determination) RBF kernel is trained on the ensemble data. The ARD kernel learns a separate lengthscale per parameter dimension: short lengthscales indicate parameters the log-likelihood is sensitive to, long lengthscales indicate parameters that have little effect. This provides a built-in sensitivity analysis.

The trained surrogate predicts both a mean and variance at any point in parameter space, providing likelihood estimates in microseconds (vs. minutes for the full solver) along with an uncertainty estimate on the approximation itself.

### Stage 4 — MCMC Posterior Sampling

The `emcee` affine-invariant ensemble sampler explores the posterior using the GP surrogate for likelihood evaluations. With 32 walkers running 5000 steps, this produces ~128,000 posterior samples (after burn-in and thinning) in seconds. Diagnostics include autocorrelation times, effective sample size, and acceptance fractions to verify convergence.

### Posterior Analysis

The posterior samples support:

- **Summary statistics**: calibrated mean, standard deviation, and 95% credible interval for each coefficient
- **Shift from prior**: how many prior standard deviations each parameter moved, indicating which coefficients the data is informative about
- **Corner plots**: joint and marginal distributions showing inter-parameter correlations
- **Posterior predictive checks**: running the actual CFD solver at a subset of posterior samples to validate predictions without surrogate error

## What's Implemented

- **SST k-omega turbulence model** with all 11 Menter coefficients
- **SIMPLE pressure-velocity coupling** solver for incompressible RANS
- **Finite volume method** with Green-Gauss gradients, first-order upwind convection, and central differencing for diffusion
- **Linear solvers**: AMG (pressure), BiCGSTAB (momentum/turbulence), PCG, Gauss-Seidel
- **Observation operator** mapping CFD fields to experimental quantities (drag, skin friction, velocity profiles, pressure taps, separation point)
- **Forward model** orchestration with warm-start caching, convergence classification, and MCMC-ready log-likelihood evaluation
- **Inference parameter system** with configurable presets for which SST constants to calibrate (2, 4, or all 11)
- **Validation cases**: channel flow (Re_b = 6800 vs Dean correlation) and flat plate (Re_L = 2e6 vs Schoenherr correlation)
- **Python bindings** via pybind11
- **Bayesian inference layer**: GP surrogate, emcee MCMC sampler, posterior diagnostics
- **Flow field visualization**

### Areas for Expansion
- Adjoint-based MCMC
- Stochastic discrepancy term
- Compressible/Heated flow
- Transient RANS

## Prerequisites

To build and run this project, you will need the following installed on your system:

- **C++ Compiler:** A compiler supporting C++17 or later (e.g., GCC 9+, Clang 10+, Apple Clang).
- **CMake:** Version 3.15 or newer.
- **Build System:** `make`
- **Python:** Python 3.8 or newer (required for the Bayesian inference pipeline and plotting).
- **Python Packages:** `numpy`, `scipy`, `matplotlib`, `emcee`, and `scikit-learn` (for the GP surrogate). 
- **pybind11:** Required for Python bindings (often installed via pip or system package manager).

## Building

```bash
mkdir build && cd build
cmake ..
make
```

### C++ Validation

```bash
./rans_sst --validate-channel    # Channel flow validation
./rans_sst --validate-plate      # Flat plate validation
./rans_sst --demo                # Forward model demo
./rans_sst --all                 # Run all
```