"""
RANS-SST Bayesian Inference Wrapper

Calibrates SST k-omega turbulence model coefficients against experimental
or DNS observables using Gaussian process surrogate-accelerated MCMC.
Designed to interface with the C++ forward model via pybind11 bindings
(rans_sst_py module).

Usage: 
    import rans_sst_py as rs
"""

import rans_sst_py as rs

import numpy as np
import importlib.util
from typing import Optional, Tuple, List, Dict
import time

import GPy
import emcee
import corner

#  Prior Class: Truncated Normal centered on Menter (1994) defaults

class Prior: 
    """
    Truncated Normal prior over SST closure coefficients.

    Each parameter θ_i has:
      - mean                μ_i   (Menter 1994 default)
      - standard deviation  σ_i   (default: 15% of mean)
      - bounds      [lo_i, hi_i]  (physical positivity + stability constraints)

    log p(θ) = -0.5 * Σ_i ((θ_i - μ_i) / σ_i)^2   if all θ_i ∈ [lo_i, hi_i]
             = -inf                               otherwise
    """

    def __init__(self, means, stds, lower, upper):
        self.means = np.asarray(means, dtype=np.float64)
        self.stds  = np.asarray(stds,  dtype=np.float64)
        self.lower = np.asarray(lower, dtype=np.float64)
        self.upper = np.asarray(upper, dtype=np.float64)
        self.ndim  = len(self.means)

    def log_prior(self, theta):
        """Evaluate log-prior at parameter vector theta."""
        theta = np.asarray(theta)
        if np.any(theta < self.lower) or np.any(theta > self.upper):
            return -np.inf
        if np.any(~np.isfinite(theta)):
            return -np.inf
        z = (theta - self.means) / self.stds
        return -0.5 * np.sum(z * z)
    
    def sample(self, n=1):
        """
        Draw n samples from the truncated normal prior via rejection.
        Rejection sampling is efficient low-dimensional parameter spaces (2-11 dims) 
        since the truncation region is wide relative to the prior width.
        """
        samples = np.empty((n, self.ndim))
        for i in range(n):
            for j in range(self.ndim):
                while True:
                    s = np.random.normal(self.means[j], self.stds[j])
                    if self.lower[j] <= s <= self.upper[j]:
                        samples[i, j] = s
                        break
        return samples
    
def make_prior_from_param_set(param_set, relative_std=0.15):
    """
    Build a Prior from an InferenceParameterSet.
    Accepts either a C++ pybind11 object (has .pack() method) or
    a plain dict with keys 'defaults', 'lower', 'upper' for
    standalone testing without C++ bindings.

    Parameters
    param_set: InferenceParameterSet or dict
        Parameter set defining active SST coefficients.
    relative_std : float
        Prior standard deviation as fraction of mean (default 15%).

    Returns: Prior
    """
    if hasattr(param_set, 'pack'):
        # C++ pybind11 InferenceParameterSet
        defaults = param_set.pack(rs.SSTCoefficients())
        lo = param_set.lower_bounds()
        hi = param_set.upper_bounds()
    else:
        # dict fallback for standalone testing
        defaults = param_set['defaults']
        lo = param_set['lower']
        hi = param_set['upper']

    means = np.array(defaults)
    stds = np.maximum(relative_std * np.abs(means), 1e-6)
    return Prior(means, stds, np.array(lo), np.array(hi))


# Latin Hypercube Sampling

def latin_hypercube(n, ndim, lower, upper):
    """
    Stratified Latin hypercube in [lower, upper]^ndim.

    Divides each dimension into n equal strata and places exactly one
    sample per stratum, ensuring uniform marginal coverage.  This gives
    much better space-filling than pure random sampling for the same
    budget, which matters when each evaluation is a full CFD solve.

    Parameters:
    n: int
        Number of samples (= number of CFD evaluations in ensemble).
    ndim: int
        Dimensionality (= number of active SST parameters)
    lower, upper : array-like, shape (ndim,)
        Physical bounds for each parameter

    Returns
    samples: ndarray, shape (n, ndim)
    """
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    samples = np.empty((n, ndim))
    for j in range(ndim):
        perm = np.random.permutation(n)
        for i in range(n):
            lo = lower[j] + (upper[j] - lower[j]) * perm[i] / n
            hi = lower[j] + (upper[j] - lower[j]) * (perm[i] + 1) / n
            samples[i, j] = np.random.uniform(lo, hi)
    return samples


# Gaussian Process Surrogate

class GPSurrogate:
    """
    Gaussian process surrogate for the forward model log-likelihood surface.

    Trained on (θ, loglik) pairs from the CFD ensemble.  Once trained,
    provides cheap (μ, σ²) predictions at any θ, enabling MCMC with
    ~10^4 posterior evaluations at negligible cost.

    Uses GPy with ARD (automatic relevance determination) RBF kernel
    to automatically identify which SST parameters dominate the
    log-likelihood variation.
    """

    def __init__(self):
        self.gp = None
        self.X_train = None
        self.y_train = None
        self.trained = False
        self._train_time = 0.0

    def train(self, X, y, optimize_restarts=3):
        """
        Train GP on ensemble data

        Parameters
        X: ndarray, shape (n_train, ndim)
            Parameter vectors from ensemble
        y: ndarray, shape (n_train,)
            Log-likelihood values from forward model
        optimize_restarts : int
            Number of random restarts for kernel hyperparameter optimization
            More restarts reduce risk of local optima in marginal likelihood
        """
        self.X_train = X.copy()
        self.y_train = y.copy().reshape(-1, 1)

        ndim = X.shape[1]
        kernel = GPy.kern.RBF(ndim, ARD=True)

        t0 = time.time()
        self.gp = GPy.models.GPRegression(X, self.y_train, kernel)
        self.gp.optimize_restarts(
            num_restarts=optimize_restarts,
            messages=False, verbose=False
        )
        self._train_time = time.time() - t0
        self.trained = True

    def predict(self, theta):
        """
        Predict (mean, variance) at a single parameter vector
        Returns
        mu: float
            Predicted log-likelihood
        var: float
            Predictive variance - uncertainty in surrogate prediction
        """
        assert self.trained, "Surrogate not trained"
        X = np.asarray(theta).reshape(1, -1)
        mu, var = self.gp.predict(X)
        return float(mu[0, 0]), float(var[0, 0])

    def predict_batch(self, Theta):
        """Predict (means, variances) for a batch of parameter vectors"""
        assert self.trained, "Surrogate not trained"
        Theta = np.atleast_2d(Theta)
        mu, var = self.gp.predict(Theta)
        return mu.ravel(), var.ravel()

    def log_likelihood(self, theta):
        """Surrogate log-likelihood (mean prediction, ignoring variance)"""
        mu, _ = self.predict(theta)
        return mu

    def rmse(self, X_test, y_test):
        """Root mean squared error on holdout set"""
        mu, _ = self.predict_batch(X_test)
        return float(np.sqrt(np.mean((mu - y_test) ** 2)))

    def lengthscales(self):
        """
        ARD lengthscales from the RBF kernel.

        Short lengthscale -> log-likelihood varies rapidly with that parameter(high sensitivity) 
        Long lengthscale -> parameter has little effect
        Useful for identifying which SST coefficients actually matter
        """
        if not self.trained:
            return None
        return self.gp.kern.lengthscale.values.copy()



