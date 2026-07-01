"""
rlf_fit.py
==========
Binned radio luminosity function from the stochastic AGN model, plus chi^2
fitting against the Kondapally+22 RLF. Works for BOTH duty-cycle models
(constant and power-law) through generic parameter handling.

Key fix vs the notebook: empty luminosity bins are returned as NaN instead of
being floored to ~1e-308 and then log10'd to ~-300, which used to detonate the
chi^2. The chi^2 simply masks non-finite bins.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import radiogalaxies_duty_ref as rgd


# ----------------------------------------------------------------------------
# Observed RLF (Kondapally+22): load + package
# ----------------------------------------------------------------------------
def load_kondapally(path, binwidth=0.3, pad=0.15):
    """Read a Kondapally machine-readable LERG LF file into an obs dict.

    Columns 'rho_lerg' are log10(rho / Mpc^-3 dex^-1); errors are in the same
    logged units. Returns the bin grid matched to the table spacing.
    """
    df = pd.read_csv(path, skipfooter=3, engine="python", dtype=str)
    df = df.apply(pd.to_numeric, errors="ignore")
    logL = df["logL"].to_numpy()
    lum_min, lum_max = logL.min() - pad, logL.max() + pad
    return {
        "logL": logL,
        "y": df["rho_lerg"].to_numpy(),            # log10(rho)
        "yerr_up": df["rho_lerg_perr"].to_numpy(),
        "yerr_down": df["rho_lerg_nerr"].to_numpy(),
        "lum_min": lum_min,
        "lum_max": lum_max,
        "n_bins": int(round((lum_max - lum_min) / binwidth)),
    }


# ----------------------------------------------------------------------------
# chi^2 (asymmetric errors), masking non-finite model bins
# ----------------------------------------------------------------------------
def chi2_asym(y_pred, y, yerr_up, yerr_down):
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_pred) & np.isfinite(y)
    if not mask.any():
        return np.inf
    yp, yo = y_pred[mask], np.asarray(y)[mask]
    eu, ed = np.asarray(yerr_up)[mask], np.asarray(yerr_down)[mask]
    over = yp >= yo
    return float(np.sum(np.where(over, ((yp - yo) / eu) ** 2, ((yp - yo) / ed) ** 2)))


# ----------------------------------------------------------------------------
# Mean binned model RLF over n_resim realisations
# ----------------------------------------------------------------------------
def rlf_binned(sim_data, V, lum_min, lum_max, n_bins, *,
               n_resim=64, seed=42, logMcut=10.0, **model_kwargs):
    """Mean log10(rho) per dex over n_resim seeds. **model_kwargs are passed
    straight to stochastic_agn_catalog -> set EITHER duty_cycle=... (constant)
    OR duty_D0=..., duty_beta=... (power-law), plus lognormR, alphaR, etc."""
    edges = np.linspace(lum_min, lum_max, n_bins + 1)
    dlog = np.diff(edges)
    hists = np.empty((n_resim, n_bins))
    for j, s in enumerate(range(seed, seed + n_resim)):
        res = rgd.stochastic_agn_catalog(data=sim_data, seed=s, logMcut=logMcut,
                                         V=V, **model_kwargs)
        hists[j], _ = np.histogram(res["logL"], bins=edges)
    mean_counts = hists.mean(axis=0)
    with np.errstate(divide="ignore"):
        log_rho = np.where(mean_counts > 0, np.log10(mean_counts / V / dlog), np.nan)
    centers = 0.5 * (edges[1:] + edges[:-1])
    return centers, log_rho


# ----------------------------------------------------------------------------
# Generic chi^2 fit: name the free parameters, fix the rest
# ----------------------------------------------------------------------------
def make_cost(sim_data, V, obs, param_names, fixed=None, *,
              n_resim=32, seed=42, logMcut=10.0):
    fixed = dict(fixed or {})

    def cost(theta):
        kw = dict(zip(param_names, theta))
        kw.update(fixed)
        _, y_pred = rlf_binned(sim_data, V, obs["lum_min"], obs["lum_max"],
                               obs["n_bins"], n_resim=n_resim, seed=seed,
                               logMcut=logMcut, **kw)
        return chi2_asym(y_pred, obs["y"], obs["yerr_up"], obs["yerr_down"])

    return cost


def fit_model(sim_data, V, obs, param_names, p0, fixed=None, *,
              n_resim=32, seed=42, logMcut=10.0, method="Nelder-Mead"):
    """Minimise chi^2. Returns (scipy_result, best_kwargs, chi2_per_dof)."""
    cost = make_cost(sim_data, V, obs, param_names, fixed,
                     n_resim=n_resim, seed=seed, logMcut=logMcut)
    res = minimize(cost, np.asarray(p0, dtype=float), method=method)
    best = dict(zip(param_names, res.x))
    best.update(fixed or {})
    ndof = max(len(obs["y"]) - len(param_names), 1)
    return res, best, res.fun / ndof


# Convenience presets for the two models -------------------------------------
def fit_constant(sim_data, V, obs, p0=(0.1, 24.0, 1.3),
                 sigma_logL=0.5, logmnorm=11.0, **kw):
    return fit_model(sim_data, V, obs,
                     param_names=["duty_cycle", "lognormR", "alphaR"], p0=p0,
                     fixed={"sigma_logL": sigma_logL, "logmnorm": logmnorm}, **kw)


def fit_powerlaw(sim_data, V, obs, p0=(0.1, 1.0, 24.0, 1.3),
                 duty_pivot=11.0, sigma_logL=0.5, logmnorm=11.0, **kw):
    """Free: (duty_D0, duty_beta, lognormR, alphaR). duty_pivot fixed."""
    return fit_model(sim_data, V, obs,
                     param_names=["duty_D0", "duty_beta", "lognormR", "alphaR"],
                     p0=p0,
                     fixed={"duty_pivot": duty_pivot, "sigma_logL": sigma_logL,
                            "logmnorm": logmnorm}, **kw)
