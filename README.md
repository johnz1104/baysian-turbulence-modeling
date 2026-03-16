# Bayesian Turbulence Modeling

A framework for Bayesian calibration of RANS turbulence model constants using experimental data.

## Motivation

RANS turbulence models like Menter's SST k-omega rely on empirical constants that were tuned for general-purpose flows. For specific flow configurations, these default values may not be optimal. This project uses Bayesian inference to learn improved parameter values from experimental measurements, producing not just point estimates but full posterior distributions that quantify parametric uncertainty.


### What's implemented

- **SST k-omega turbulence model** with all 11 Menter coefficients
- **SIMPLE pressure-velocity coupling** solver for incompressible RANS
- **Finite volume method** with Green-Gauss gradients, first-order upwind convection, and central differencing for diffusion
- **Linear solvers**: AMG (pressure), BiCGSTAB (momentum/turbulence), PCG, Gauss-Seidel
- **Observation operator** mapping CFD fields to experimental quantities (drag, skin friction, velocity profiles, pressure taps, separation point)
- **Forward model** orchestration with warm-start caching, convergence classification, and MCMC-ready log-likelihood evaluation
- **Inference parameter system** with configurable presets for which SST constants to calibrate (2, 4, or all 11)
- **Validation cases**: channel flow (Re_b = 6800 vs Dean correlation) and flat plate (Re_L = 2e6 vs Schoenherr correlation)
- **Python bindings** via pybind11
- **Bayesian inference layer** MCMC sampler
- **Flow field visualization**

### Planned
- Adjoint-based MCMC for gradient-informed proposals


## Building

```bash
mkdir build && cd build
cmake ..
make
```

## Usage

```bash
./rans_sst --validate-channel    # Channel flow validation
./rans_sst --validate-plate      # Flat plate validation
./rans_sst --demo                # Forward model demo
./rans_sst --all                 # Run all
```
