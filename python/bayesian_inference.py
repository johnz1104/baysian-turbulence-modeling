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



# Bayesian Inference Pipeline

class BayesianInference:
    """
    Full Bayesian calibration pipeline for SST turbulence coefficients.

    Orchestrates: prior specification -> ensemble generation -> GP surrogate training -> MCMC sampling -> posterior diagnostics.

    Parameters
    forward_model: object
        C++ ForwardModel (via pybind11) or any callable with
        penalized_log_likelihood(theta) -> float.
    param_set: object
        InferenceParameterSet — defines which SST coefficients are active.
    prior: Prior, optional
        If None, constructs truncated normal prior from param_set defaults.
    """

    def __init__(self, forward_model, param_set, prior=None):
        self.forward_model = forward_model
        self.param_set = param_set
        self.prior = prior if prior is not None else make_prior_from_param_set(param_set)
        self.surrogate = GPSurrogate()

        # ensemble storage
        self.ensemble_X = None      # shape (n_valid, ndim)
        self.ensemble_y = None      # shape (n_valid,)
        self.ensemble_status = None # list of EvaluationStatus per run

        # MCMC storage
        self.samples = None         # shape (n_samples, ndim)
        self.sampler = None         # emcee sampler object (for diagnostics)

    # Ensemble generation
    def run_ensemble(self, n_samples=200, verbose=True):
        """
        Generate Latin hypercube design and evaluate forward model.

        Each evaluation runs the full C++ pipeline:
          θ -> unpack SST coeffs -> SIMPLE solve (AMG pressure, BiCGSTAB momentum)
            -> observation operator H(·) -> Gaussian log-likelihood

        Failed evaluations (diverged, invalid parameters) are filtered
        out before surrogate training, but their count is reported.

        Parameters
        n_samples: int
            Number of ensemble members. 200 is typical for 2-4 active parameters; increase to 500+ for all 11 preset.
        verbose: bool
            Print progress every 20 evaluations.

        Returns
        X: ndarray, shape (n_valid, ndim)
        y: ndarray, shape (n_valid,)
        """
        ndim = self.prior.ndim
        X = latin_hypercube(n_samples, ndim, self.prior.lower, self.prior.upper)
        y = np.full(n_samples, -np.inf)
        statuses = []

        t0 = time.time()
        for i in range(n_samples):
            theta_list = X[i].tolist()

            # use full evaluate() if available to get status info
            if hasattr(self.forward_model, 'evaluate'):
                result = self.forward_model.evaluate(theta_list)
                y[i] = result.log_lik
                statuses.append(str(result.status))
            else:
                y[i] = self.forward_model.penalized_log_likelihood(theta_list)
                statuses.append("Unknown")

            if verbose and (i + 1) % 20 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                print(f"  Ensemble {i+1}/{n_samples}  "
                      f"loglik={y[i]:.4f}  "
                      f"[{rate:.1f} eval/s]")

        # filter out failed evaluations (penalty values)
        valid = y > -1e5
        self.ensemble_X = X[valid]
        self.ensemble_y = y[valid]
        self.ensemble_status = statuses

        n_valid = int(np.sum(valid))
        n_diverged = n_samples - n_valid
        elapsed = time.time() - t0

        if verbose:
            print(f"\n  Ensemble complete: {n_valid}/{n_samples} valid "
                  f"({n_diverged} diverged/invalid)  "
                  f"[{elapsed:.1f}s total]")

        return self.ensemble_X, self.ensemble_y


    # Surrogate training

    def train_surrogate(self, holdout_frac=0.1, verbose=True):
        """
        Train GP surrogate on ensemble data with holdout validation.

        Splits ensemble into training and holdout sets, trains ARD-RBF
        GP, reports holdout RMSE and kernel lengthscales.

        Parameters
        holdout_frac: float
            Fraction of ensemble reserved for validation (default 10%).
        verbose: bool
            Print training summary.

        Returns
        rmse : float
            Holdout RMSE (units of log-likelihood).
        """
        assert self.ensemble_X is not None, \
            "No ensemble data — call run_ensemble() first"

        n = len(self.ensemble_X)
        assert n >= 10, \
            f"Only {n} valid ensemble points — need at least 10"

        n_test = max(1, int(n * holdout_frac))
        idx = np.random.permutation(n)

        X_train = self.ensemble_X[idx[n_test:]]
        y_train = self.ensemble_y[idx[n_test:]]
        X_test  = self.ensemble_X[idx[:n_test]]
        y_test  = self.ensemble_y[idx[:n_test]]

        self.surrogate.train(X_train, y_train)
        rmse = self.surrogate.rmse(X_test, y_test)

        if verbose:
            print(f"  Surrogate trained: {len(X_train)} train, "
                  f"{n_test} holdout")
            print(f"  Holdout RMSE:      {rmse:.4f}")
            print(f"  Training time:     {self.surrogate._train_time:.2f}s")

            ls = self.surrogate.lengthscales()
            if ls is not None:
                names = _get_param_names(self.param_set)
                print(f"  ARD lengthscales:")
                for name, l in zip(names, ls):
                    print(f"    {name:12s}  {l:.4f}")

        return rmse

    
    # Log-posterior for MCMC

    def log_posterior(self, theta):
        """
        log p(θ|y) ∝ log p(θ) + log p(y|θ)

        Prior: truncated normal on SST coefficients.
        Likelihood: GP surrogate prediction (mean only — ignoring predictive variance is standard in surrogate-accelerated MCMC).
        """
        lp = self.prior.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        ll = self.surrogate.log_likelihood(theta)
        if not np.isfinite(ll):
            return -np.inf
        return lp + ll


    #  MCMC sampling

    def run_mcmc(self, n_walkers=32, n_steps=5000, burn_in=1000,
                 thin=1, verbose=True):
        """
        Run emcee affine-invariant ensemble sampler.

        Walkers are initialised near the prior mean with small random
        perturbations.  The sampler uses the GP surrogate for likelihood
        evaluations (~μs per call instead of ~minutes for full CFD).

        Parameters
        n_walkers: int
            Number of walkers (must be >= 2 * ndim).
        n_steps: int
            Total MCMC steps per walker.
        burn_in: int
            Steps to discard as burn-in.
        thin: int
            Thinning factor for final samples.
        verbose: bool
            Print progress and diagnostics.

        Returns
        samples: ndarray, shape (n_effective, ndim)
            Posterior samples after burn-in and thinning.
        """

        assert self.surrogate.trained, \
            "Surrogate not trained"

        ndim = self.prior.ndim
        if n_walkers < 2 * ndim:
            n_walkers = 2 * ndim
            if verbose:
                print(f"  Increased n_walkers to {n_walkers} (need >= 2*ndim)")

        # initialise walkers: small perturbation around prior mean
        p0 = np.empty((n_walkers, ndim))
        for i in range(n_walkers):
            p0[i] = self.prior.means + 0.01 * self.prior.stds * np.random.randn(ndim)
            p0[i] = np.clip(p0[i], self.prior.lower, self.prior.upper)

        self.sampler = emcee.EnsembleSampler(n_walkers, ndim, self.log_posterior)

        if verbose:
            print(f"  MCMC: {n_walkers} walkers x {n_steps} steps "
                  f"(burn-in={burn_in}, thin={thin})")

        t0 = time.time()
        self.sampler.run_mcmc(p0, n_steps, progress=verbose)
        elapsed = time.time() - t0

        self.samples = self.sampler.get_chain(
            discard=burn_in, thin=thin, flat=True
        )

        if verbose:
            print(f"\n  MCMC complete: {len(self.samples)} posterior samples "
                  f"[{elapsed:.1f}s]")
            self._print_diagnostics()

        return self.samples

    def _print_diagnostics(self):
        """Print MCMC convergence diagnostics."""
        # emcee's get_autocorr_time requires chains long enough for
        # a reliable estimate — roughly 50x the autocorrelation time.
        # for short chains it raises AutocorrError.  we check the chain
        # length heuristic (> 50 * ndim) before attempting the estimate.
        chain = self.sampler.get_chain()
        n_steps = chain.shape[0]
        ndim = chain.shape[2]

        # heuristic: need at least 50 steps per dimension for a
        # meaningful autocorrelation estimate
        if n_steps >= 50 * ndim:
            tau = self.sampler.get_autocorr_time(quiet=True)
            if np.all(np.isfinite(tau)):
                names = _get_param_names(self.param_set)
                print(f"  Autocorrelation times:")
                for name, t in zip(names, tau):
                    print(f"    {name:12s}  τ = {t:.1f}")
                n_eff = len(self.samples) / np.max(tau)
                print(f"  Effective samples:  ~{int(n_eff)}")
            else:
                print("  (Autocorrelation times contain non-finite values — "
                      "chain may need more steps)")
        else:
            print(f"  (Chain too short for autocorrelation estimate: "
                  f"{n_steps} steps, need ~{50 * ndim}+)")

        # acceptance fraction
        af = self.sampler.acceptance_fraction
        print(f"  Acceptance fraction: {np.mean(af):.3f} "
              f"(range [{np.min(af):.3f}, {np.max(af):.3f}])")

    # ----------------------------------------------------------------
    #  Posterior analysis
    # ----------------------------------------------------------------

    def posterior_summary(self):
        """
        Compute posterior statistics: mean, std, 95% credible interval.

        Returns
        -------
        summary : dict
            Keyed by parameter name.  Each entry contains:
              mean, std, ci_2.5, ci_97.5, prior_mean, shift
            where shift = (posterior_mean - prior_mean) / prior_std
            quantifies how much the data moved the parameter.
        """
        assert self.samples is not None, \
            "No posterior samples — call run_mcmc() first"

        names = _get_param_names(self.param_set)
        summary = {}

        for i, name in enumerate(names):
            s = self.samples[:, i]
            shift = (np.mean(s) - self.prior.means[i]) / self.prior.stds[i]
            summary[name] = {
                'mean':       float(np.mean(s)),
                'std':        float(np.std(s)),
                'ci_2.5':     float(np.percentile(s, 2.5)),
                'ci_97.5':    float(np.percentile(s, 97.5)),
                'prior_mean': float(self.prior.means[i]),
                'shift':      float(shift),
            }

        return summary

    def print_summary(self):
        """Print formatted posterior summary table."""
        summary = self.posterior_summary()
        names = _get_param_names(self.param_set)

        print(f"\n  {'Parameter':>12s}  {'Prior':>8s}  {'Posterior':>10s}  "
              f"{'±σ':>7s}  {'95% CI':>18s}  {'Shift':>6s}")
        print("  " + "-" * 72)

        for name in names:
            s = summary[name]
            print(f"  {name:>12s}  {s['prior_mean']:8.4f}  {s['mean']:10.4f}  "
                  f"{s['std']:7.4f}  "
                  f"[{s['ci_2.5']:.4f}, {s['ci_97.5']:.4f}]  "
                  f"{s['shift']:+6.2f}σ")

    def plot_posterior(self, save_path=None):
        """
        Corner plot of posterior with prior means marked.

        Requires the 'corner' package (pip install corner).
        Returns None if corner is not installed.
        """
        if not _HAS_CORNER:
            print("  corner package not installed — skipping plot")
            return None

        import corner
        names = _get_param_names(self.param_set)
        fig = corner.corner(
            self.samples, labels=names,
            truths=self.prior.means.tolist(),
            show_titles=True, title_fmt='.4f',
            quantiles=[0.16, 0.5, 0.84]
        )
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Corner plot saved to {save_path}")
        return fig

    # ----------------------------------------------------------------
    #  Posterior predictive check
    # ----------------------------------------------------------------

    def posterior_predictive(self, n_samples=50, verbose=True):
        """
        Run forward model at posterior samples to get predictive distribution.

        This is expensive (n_samples full CFD solves) but gives the
        actual posterior predictive uncertainty without surrogate error.
        Use for final validation, not routine analysis.

        Parameters
        ----------
        n_samples : int
            Number of posterior samples to evaluate (subset of full chain).

        Returns
        -------
        predictions : list of ndarray
            Each entry is the H(fields) vector from one posterior sample.
        """
        assert self.samples is not None, "No posterior samples"

        # subsample from posterior
        idx = np.random.choice(len(self.samples), size=n_samples, replace=False)
        predictions = []
        statuses = []

        for i, si in enumerate(idx):
            theta = self.samples[si].tolist()
            result = self.forward_model.evaluate(theta)
            predictions.append(np.array(result.predictions))
            statuses.append(str(result.status))

            if verbose and (i + 1) % 10 == 0:
                print(f"  Predictive check {i+1}/{n_samples}: "
                      f"{result.status}")

        n_converged = sum(1 for s in statuses
                          if 'Converged' in s or 'Unconverged' in s)
        if verbose:
            print(f"  Predictive check complete: "
                  f"{n_converged}/{n_samples} converged")

        return predictions
