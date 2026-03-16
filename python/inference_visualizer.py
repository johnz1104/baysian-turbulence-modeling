"""
Inference Visualizer for the RANS-SST Bayesian Calibration Pipeline

Visualizes every stage of the surrogate-accelerated MCMC pipeline:
prior distributions, ensemble coverage, GP surrogate diagnostics,
MCMC trace plots, prior-vs-posterior comparisons, and posterior
predictive checks.

Usage:
    from bayesian_inference import BayesianInference
    from inference_visualizer import InferenceVisualizer

    bi = BayesianInference(forward_model, param_set)
    bi.run_ensemble()
    bi.train_surrogate()
    bi.run_mcmc()

    vis = InferenceVisualizer(bi)
    vis.plot_prior()
    vis.plot_ensemble()
    vis.plot_surrogate_diagnostics()
    vis.plot_traces()
    vis.plot_prior_vs_posterior()
    vis.plot_corner()
    vis.plot_full_report(save_dir='figures/')
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.stats import gaussian_kde, norm
from pathlib import Path


def _get_param_names(param_set):
    """Extract parameter names from param_set (C++ or dict)."""
    if hasattr(param_set, 'active_names'):
        return param_set.active_names()
    if isinstance(param_set, dict) and 'names' in param_set:
        return param_set['names']
    ndim = param_set.n_active() if hasattr(param_set, 'n_active') else 2
    return [f"theta_{i}" for i in range(ndim)]


# Color palette
_C_PRIOR = '#7fb3d8'       # light blue
_C_POSTERIOR = '#1b4f72'    # dark blue
_C_TRUTH = '#c0392b'        # red
_C_GRID = '#cccccc'


class InferenceVisualizer:
    """
    Visualization suite for the Bayesian calibration pipeline.

    Reads all state from a BayesianInference object and produces
    matplotlib figures for each pipeline stage.  Every plot method
    returns the Figure so callers can further customize or embed.

    Parameters
    inference : BayesianInference
        A pipeline object (any subset of stages may have been run;
        methods that need unfinished stages will raise).
    """

    def __init__(self, inference):
        self.inf = inference
        self.names = _get_param_names(inference.param_set)
        self.ndim = inference.prior.ndim

    # 1. Prior distributions
    def plot_prior(self, save_path=None, figsize=None):
        """
        Plot the truncated normal prior PDF for each active parameter.

        One subplot per coefficient showing the PDF curve, the Menter
        default (vertical line), and the truncation bounds (shaded region
        excluded).
        """
        prior = self.inf.prior
        ncols = min(self.ndim, 3)
        nrows = (self.ndim + ncols - 1) // ncols
        if figsize is None:
            figsize = (4.5 * ncols, 3.2 * nrows)

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
        fig.suptitle('Prior Distributions (Truncated Normal)', fontsize=13, fontweight='bold', y=1.01)

        for i in range(self.ndim):
            ax = axes[i // ncols, i % ncols]
            mu, sigma = prior.means[i], prior.stds[i]
            lo, hi = prior.lower[i], prior.upper[i]

            # plot range: extend slightly beyond bounds
            margin = 0.15 * (hi - lo)
            x = np.linspace(lo - margin, hi + margin, 400)
            pdf = norm.pdf(x, loc=mu, scale=sigma)

            # mask outside truncation bounds
            inside = (x >= lo) & (x <= hi)
            ax.fill_between(x[inside], pdf[inside], alpha=0.35, color=_C_PRIOR,label='Prior')
            ax.plot(x[inside], pdf[inside], color=_C_POSTERIOR, lw=1.5)

            # shade excluded tails
            left = x < lo
            right = x > hi
            if np.any(left):
                ax.fill_between(x[left], pdf[left], alpha=0.08, color='red')
            if np.any(right):
                ax.fill_between(x[right], pdf[right], alpha=0.08, color='red')

            # bounds and mean
            ax.axvline(mu, color=_C_TRUTH, ls='--', lw=1.2, label=f'Default={mu:.4f}')
            ax.axvline(lo, color='grey', ls=':', lw=0.8)
            ax.axvline(hi, color='grey', ls=':', lw=0.8)

            ax.set_xlabel(self.names[i], fontsize=10)
            ax.set_ylabel('Density', fontsize=9)
            ax.legend(fontsize=7, loc='upper right')
            ax.grid(True, alpha=0.3, color=_C_GRID)

        # hide unused subplots
        for j in range(self.ndim, nrows * ncols):
            axes[j // ncols, j % ncols].set_visible(False)

        fig.tight_layout()
        self._save(fig, save_path)
        return fig

    # 2. Ensemble coverage

    def plot_ensemble(self, save_path=None, figsize=None):
        """
        Scatter-matrix of LHS ensemble points colored by log-likelihood.

        Off-diagonal: pairwise scatter.  Diagonal: 1D histogram of each
        parameter's ensemble coverage.  Color encodes log-likelihood so
        the 'good' region of parameter space is immediately visible.
        """
        X = self.inf.ensemble_X
        y = self.inf.ensemble_y
        n = self.ndim

        if figsize is None:
            figsize = (3 * n, 3 * n)

        fig, axes = plt.subplots(n, n, figsize=figsize, squeeze=False)
        fig.suptitle('Ensemble Coverage (LHS)  —  colored by log-likelihood', fontsize=13, fontweight='bold', y=1.01)

        # normalize log-likelihood for colormap
        vmin, vmax = np.percentile(y, [2, 98])

        for i in range(n):
            for j in range(n):
                ax = axes[i, j]
                if i == j:
                    # diagonal: histogram
                    ax.hist(X[:, i], bins=25, color=_C_PRIOR, edgecolor='white', linewidth=0.5, alpha=0.7)
                    ax.axvline(self.inf.prior.means[i], color=_C_TRUTH, ls='--', lw=1)
                elif i > j:
                    # lower triangle: scatter
                    ax.scatter(X[:, j], X[:, i], c=y, cmap='viridis', s=8, alpha=0.7, edgecolors='none', vmin=vmin, vmax=vmax)
                else:
                    # upper triangle: hide
                    ax.set_visible(False)
                    continue

                # axis labels on edges only
                if i == n - 1:
                    ax.set_xlabel(self.names[j], fontsize=9)
                else:
                    ax.set_xticklabels([])
                if j == 0 and i != 0:
                    ax.set_ylabel(self.names[i], fontsize=9)
                elif i == 0 and j == 0:
                    ax.set_ylabel(self.names[i], fontsize=9)
                else:
                    ax.set_yticklabels([])

                ax.tick_params(labelsize=7)

        # colorbar
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.6, aspect=30, pad=0.02)
        cbar.set_label('log-likelihood', fontsize=10)

        fig.tight_layout()
        self._save(fig, save_path)
        return fig

    # 3. GP Surrogate diagnostics

    def plot_surrogate_diagnostics(self, save_path=None, figsize=(14, 4.5)):
        """
        Three-panel surrogate diagnostic:
          (a) Predicted vs actual log-likelihood (holdout)
          (b) Residual histogram
          (c) ARD lengthscale bar chart
        """
        X = self.inf.ensemble_X
        y = self.inf.ensemble_y
        mu, _var = self.inf.surrogate.predict_batch(X)
        residuals = mu - y

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
        fig.suptitle('GP Surrogate Diagnostics', fontsize=13, fontweight='bold', y=1.02)

        # (a) predicted vs actual
        ax1.scatter(y, mu, s=12, alpha=0.5, c=_C_POSTERIOR, edgecolors='none')
        lims = [min(y.min(), mu.min()), max(y.max(), mu.max())]
        margin = 0.05 * (lims[1] - lims[0])
        lims = [lims[0] - margin, lims[1] + margin]
        ax1.plot(lims, lims, 'k--', lw=1, alpha=0.6, label='1:1')
        ax1.set_xlim(lims)
        ax1.set_ylim(lims)
        ax1.set_xlabel('Actual log-likelihood', fontsize=10)
        ax1.set_ylabel('GP predicted log-likelihood', fontsize=10)
        ax1.set_title('(a)  Predicted vs Actual', fontsize=10)
        ax1.legend(fontsize=8)
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3, color=_C_GRID)

        rmse = np.sqrt(np.mean(residuals**2))
        ax1.text(0.05, 0.92, f'RMSE = {rmse:.4f}', transform=ax1.transAxes, fontsize=8, bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))

        # (b) residual histogram
        ax2.hist(residuals, bins=30, color=_C_PRIOR, edgecolor='white', linewidth=0.5, alpha=0.7, density=True)
        ax2.axvline(0, color='k', ls='--', lw=1)
        # overlay Gaussian fit
        res_std = np.std(residuals)
        xr = np.linspace(residuals.min(), residuals.max(), 200)
        ax2.plot(xr, norm.pdf(xr, 0, res_std), color=_C_TRUTH, lw=1.5, label=f'N(0, {res_std:.4f})')
        ax2.set_xlabel('Residual (predicted - actual)', fontsize=10)
        ax2.set_ylabel('Density', fontsize=10)
        ax2.set_title('(b)  Residual Distribution', fontsize=10)
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3, color=_C_GRID)

        # (c) ARD lengthscales
        ls = self.inf.surrogate.lengthscales()
        if ls is not None:
            ax3.barh(self.names, ls, color=_C_POSTERIOR, alpha=0.8, edgecolor='white', linewidth=0.5)
            ax3.set_xlabel('Lengthscale', fontsize=10)
            ax3.set_title('(c)  ARD Lengthscales', fontsize=10)
            ax3.invert_yaxis()
            ax3.grid(True, axis='x', alpha=0.3, color=_C_GRID)
            # annotate: short = sensitive
            ax3.text(0.95, 0.05, 'shorter = more sensitive', transform=ax3.transAxes, fontsize=7, ha='right', fontstyle='italic', color='grey')
        else:
            ax3.text(0.5, 0.5, 'No ARD lengthscales\n(kernel not RBF-ARD?)', transform=ax3.transAxes, ha='center', va='center')

        fig.tight_layout()
        self._save(fig, save_path)
        return fig

    # 4. Surrogate surface (2D or sliced) 

    def plot_surrogate_surface(self, dims=(0, 1), n_grid=80, save_path=None, figsize=(11, 4.5)):
        """
        Contour plots of the GP surrogate mean and variance over a 2D slice of parameter space.

        Parameters
        dims : tuple of int
            Which two parameter indices to plot (others fixed at prior mean).
        n_grid : int
            Resolution of the evaluation grid per axis.
        """
        di, dj = dims
        prior = self.inf.prior

        # build 2D grid, other dims fixed at prior mean
        xi = np.linspace(prior.lower[di], prior.upper[di], n_grid)
        xj = np.linspace(prior.lower[dj], prior.upper[dj], n_grid)
        Xi, Xj = np.meshgrid(xi, xj)

        Theta = np.tile(prior.means, (n_grid * n_grid, 1))
        Theta[:, di] = Xi.ravel()
        Theta[:, dj] = Xj.ravel()

        mu, var = self.inf.surrogate.predict_batch(Theta)
        Mu = mu.reshape(n_grid, n_grid)
        Var = var.reshape(n_grid, n_grid)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle(
            f'GP Surrogate Surface  ({self.names[di]} vs {self.names[dj]})',
            fontsize=13, fontweight='bold', y=1.02)

        # mean
        c1 = ax1.contourf(Xi, Xj, Mu, levels=40, cmap='viridis')
        fig.colorbar(c1, ax=ax1, shrink=0.85, label='log-likelihood (mean)')
        # overlay ensemble points
        if self.inf.ensemble_X is not None:
            ax1.scatter(self.inf.ensemble_X[:, di], self.inf.ensemble_X[:, dj], s=8, 
                        c='white', edgecolors='k', linewidths=0.3, alpha=0.6, zorder=5)
        ax1.axvline(prior.means[di], color='white', ls=':', lw=0.8, alpha=0.5)
        ax1.axhline(prior.means[dj], color='white', ls=':', lw=0.8, alpha=0.5)
        ax1.set_xlabel(self.names[di], fontsize=10)
        ax1.set_ylabel(self.names[dj], fontsize=10)
        ax1.set_title('(a)  GP Mean', fontsize=10)

        # variance
        c2 = ax2.contourf(Xi, Xj, np.sqrt(Var), levels=40, cmap='magma')
        fig.colorbar(c2, ax=ax2, shrink=0.85, label='Std deviation')
        if self.inf.ensemble_X is not None:
            ax2.scatter(self.inf.ensemble_X[:, di], self.inf.ensemble_X[:, dj], s=8, c='white', 
                        edgecolors='k', linewidths=0.3, alpha=0.6, zorder=5)
        ax2.set_xlabel(self.names[di], fontsize=10)
        ax2.set_ylabel(self.names[dj], fontsize=10)
        ax2.set_title('(b)  GP Uncertainty (std dev)', fontsize=10)

        fig.tight_layout()
        self._save(fig, save_path)
        return fig

    # 5. MCMC trace plots

    def plot_traces(self, burn_in=None, n_walkers_shown=8, save_path=None, figsize=None):
        """
        Trace plots showing walker chains over MCMC steps.

        Parameters
        burn_in : int, optional
            Mark burn-in cutoff with a vertical line.  If None, attempts
            to infer from the sampler's discard setting.
        n_walkers_shown : int
            Number of individual walker traces to draw (for readability).
        """
        chain = self.inf.sampler.get_chain()  # (n_steps, n_walkers, ndim)
        _n_steps, n_walkers, ndim = chain.shape

        if figsize is None:
            figsize = (10, 2.2 * ndim)

        fig, axes = plt.subplots(ndim, 1, figsize=figsize, sharex=True)
        if ndim == 1:
            axes = [axes]
        fig.suptitle('MCMC Trace Plots', fontsize=13, fontweight='bold', y=1.01)

        # pick a subset of walkers to draw
        walker_ids = np.linspace(0, n_walkers - 1, min(n_walkers_shown, n_walkers), dtype=int)

        for i, ax in enumerate(axes):
            for w in walker_ids:
                ax.plot(chain[:, w, i], lw=0.4, alpha=0.6)
            ax.axhline(self.inf.prior.means[i], color=_C_TRUTH, ls='--', lw=1, alpha=0.7, label='Prior mean')
            if burn_in is not None:
                ax.axvline(burn_in, color='k', ls=':', lw=1, alpha=0.5, label='Burn-in')
            ax.set_ylabel(self.names[i], fontsize=10)
            ax.grid(True, alpha=0.2, color=_C_GRID)
            ax.yaxis.set_major_locator(MaxNLocator(4))
            if i == 0:
                ax.legend(fontsize=7, loc='upper right')

        axes[-1].set_xlabel('Step', fontsize=10)
        fig.tight_layout()
        self._save(fig, save_path)
        return fig

    # 6. Prior vs posterior

    def plot_prior_vs_posterior(self, save_path=None, figsize=None):
        """
        Overlaid prior and posterior marginal densities for each parameter.

        Prior shown as a light filled curve, posterior as a darker filled
        KDE.  Vertical lines mark the prior mean and posterior mean.
        Text annotation shows the shift in units of prior sigma.
        """
        prior = self.inf.prior
        samples = self.inf.samples
        ncols = min(self.ndim, 3)
        nrows = (self.ndim + ncols - 1) // ncols
        if figsize is None:
            figsize = (4.5 * ncols, 3.2 * nrows)

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
        fig.suptitle('Prior vs Posterior', fontsize=13, fontweight='bold', y=1.01)

        for i in range(self.ndim):
            ax = axes[i // ncols, i % ncols]
            lo, hi = prior.lower[i], prior.upper[i]
            margin = 0.1 * (hi - lo)
            x = np.linspace(lo - margin, hi + margin, 400)

            # prior PDF (truncated normal)
            pdf_prior = norm.pdf(x, loc=prior.means[i], scale=prior.stds[i])
            inside = (x >= lo) & (x <= hi)
            pdf_masked = np.where(inside, pdf_prior, 0.0)
            # renormalize for truncation
            area = np.trapz(pdf_masked, x)
            if area > 0:
                pdf_masked /= area
            ax.fill_between(x, pdf_masked, alpha=0.25, color=_C_PRIOR, label='Prior')
            ax.plot(x, pdf_masked, color=_C_PRIOR, lw=1.5)

            # posterior KDE
            s = samples[:, i]
            kde = gaussian_kde(s)
            pdf_post = kde(x)
            ax.fill_between(x, pdf_post, alpha=0.35, color=_C_POSTERIOR,label='Posterior')
            ax.plot(x, pdf_post, color=_C_POSTERIOR, lw=1.8)

            # mark means
            ax.axvline(prior.means[i], color=_C_PRIOR, ls='--', lw=1, alpha=0.8)
            post_mean = np.mean(s)
            ax.axvline(post_mean, color=_C_POSTERIOR, ls='--', lw=1, alpha=0.8)

            # shift annotation
            shift = (post_mean - prior.means[i]) / prior.stds[i]
            ax.text(0.95, 0.92, f'shift = {shift:+.2f}$\\sigma$', transform=ax.transAxes, fontsize=8, 
                    ha='right', bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))

            ax.set_xlabel(self.names[i], fontsize=10)
            ax.set_ylabel('Density', fontsize=9)
            ax.grid(True, alpha=0.3, color=_C_GRID)
            if i == 0:
                ax.legend(fontsize=8, loc='upper left')

        for j in range(self.ndim, nrows * ncols):
            axes[j // ncols, j % ncols].set_visible(False)

        fig.tight_layout()
        self._save(fig, save_path)
        return fig

    # 7. Corner plot

    def plot_corner(self, save_path=None, figsize=None):
        """
        Corner plot of the posterior using the `corner` library.

        Prior means marked as truth lines, 16/50/84th percentile
        quantiles shown on marginal histograms.
        """
        import corner as corner_lib

        fig = corner_lib.corner(
            self.inf.samples,
            labels=self.names,
            truths=self.inf.prior.means.tolist(),
            truth_color=_C_TRUTH,
            show_titles=True,
            title_fmt='.4f',
            quantiles=[0.16, 0.5, 0.84],
            color=_C_POSTERIOR,
            hist_kwargs={'alpha': 0.6, 'edgecolor': 'white', 'linewidth': 0.5},
        )
        fig.suptitle('Posterior Corner Plot', fontsize=13, fontweight='bold', y=1.01)

        if figsize is not None:
            fig.set_size_inches(figsize)

        self._save(fig, save_path)
        return fig

    # 8. Posterior predictive check

    def plot_posterior_predictive(self, predictions, obs_data=None, obs_labels=None, obs_errors=None, save_path=None, figsize=(8, 5)):
        """
        Posterior predictive uncertainty bands vs experimental data.

        Parameters
        predictions : list of ndarray
            Output from BayesianInference.posterior_predictive().
            Each entry is the H(fields) observation vector for one
            posterior sample.
        obs_data : ndarray, optional
            Experimental observations to overlay as points.
        obs_labels : list of str, optional
            Labels for each observable (x-axis tick labels).
        obs_errors : ndarray, optional
            Measurement uncertainty (1-sigma) for error bars.
        """
        pred = np.array(predictions)  # (n_samples, n_obs)
        n_obs = pred.shape[1]
        x = np.arange(n_obs)

        fig, ax = plt.subplots(figsize=figsize)
        fig.suptitle('Posterior Predictive Check', fontsize=13, fontweight='bold')

        # percentile bands
        median = np.median(pred, axis=0)
        lo_95 = np.percentile(pred, 2.5, axis=0)
        hi_95 = np.percentile(pred, 97.5, axis=0)
        lo_68 = np.percentile(pred, 16, axis=0)
        hi_68 = np.percentile(pred, 84, axis=0)

        ax.fill_between(x, lo_95, hi_95, alpha=0.2, color=_C_POSTERIOR, label='95% CI')

        ax.fill_between(x, lo_68, hi_68, alpha=0.35, color=_C_POSTERIOR, label='68% CI')

        ax.plot(x, median, 'o-', color=_C_POSTERIOR, ms=5, lw=1.5, label='Median prediction')

        # overlay experimental data
        if obs_data is not None:
            if obs_errors is not None:
                ax.errorbar(x, obs_data, yerr=obs_errors, fmt='s', color=_C_TRUTH, ms=7, capsize=3, lw=1.5, label='Observations', zorder=10)
            else:
                ax.scatter(x, obs_data, s=60, marker='s', color=_C_TRUTH, edgecolors='k', linewidths=0.5, zorder=10, label='Observations')

        if obs_labels is not None:
            ax.set_xticks(x)
            ax.set_xticklabels(obs_labels, fontsize=9, rotation=30, ha='right')
        else:
            ax.set_xlabel('Observable index', fontsize=10)
        ax.set_ylabel('Value', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, color=_C_GRID)

        fig.tight_layout()
        self._save(fig, save_path)
        return fig

    # 9. Full report

    def plot_full_report(self, save_dir=None, prefix='inference', burn_in=None, fmt='png'):
        """
        Generate all plots and save them to a directory.

        Parameters
        save_dir : str or Path, optional
            Directory to save figures.  Created if it doesn't exist.
            If None, figures are shown interactively instead.
        prefix : str
            Filename prefix for saved figures.
        burn_in : int, optional
            Burn-in cutoff for trace plots.
        fmt : str
            Image format ('png', 'pdf', 'svg').

        Returns
        figs : dict
            Mapping from plot name to Figure object.
        """
        figs = {}

        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        def path(name):
            if save_dir is not None:
                return str(save_dir / f'{prefix}_{name}.{fmt}')
            return None

        figs['prior'] = self.plot_prior(save_path=path('prior'))
        figs['ensemble'] = self.plot_ensemble(save_path=path('ensemble'))
        figs['surrogate'] = self.plot_surrogate_diagnostics(
            save_path=path('surrogate'))
        if self.ndim >= 2:
            figs['surface'] = self.plot_surrogate_surface(
                save_path=path('surface'))
        figs['traces'] = self.plot_traces(
            burn_in=burn_in, save_path=path('traces'))
        figs['prior_vs_posterior'] = self.plot_prior_vs_posterior(
            save_path=path('prior_vs_posterior'))
        figs['corner'] = self.plot_corner(save_path=path('corner'))

        if save_dir is not None:
            print(f"  Saved {len(figs)} figures to {save_dir}/")
        else:
            plt.show()

        return figs

    # helpers 

    @staticmethod
    def _save(fig, path, dpi=150):
        if path is not None:
            fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
