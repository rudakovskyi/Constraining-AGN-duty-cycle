"""
radiogalaxies_duty.py
=====================
Physical model for LERG radio luminosities + AGN duty cycle on a galaxy/halo
catalogue.

Luminosity assignment (per active host):
    log L = lognormR + alphaR * (log M_star - logmnorm) + N(0, sigma_logL)
    i.e. a power-law mean L(M) with log-normal (Gaussian-in-logL) scatter.

Duty cycle = probability that an eligible host is radio-on. Two models:
    constant   :  f(M) = D                              (random thinning)
    power-law  :  f(M) = clip( D0 * (M/M_pivot)^beta , 0, 1 )
                  mass-dependent, CAPPED AT 1 (100% of hosts on above the
                  saturation mass log M_sat = duty_pivot - log10(D0)/beta).

A general `p_active` hook is also provided so you can later pass an arbitrary
per-host probability, e.g. an environment-dependent f(M, delta_env).

Catalogue column convention (override with `cols`):
    0 = log10(stellar mass), 1 = x, 2 = y, 3 = z   (positions in Mpc/h)
"""

import numpy as np


# ----------------------------------------------------------------------------
# Duty-cycle laws  (return a per-host probability in [0, 1])
# ----------------------------------------------------------------------------
def duty_constant(logM, D):
    """Constant duty cycle (still clipped to a valid probability)."""
    return np.clip(np.full(np.shape(logM), float(D)), 0.0, 1.0)


def duty_powerlaw(logM, D0, beta, logM_pivot=11.0):
    """Mass-dependent duty cycle f(M) = min(1, D0 * (M/M_pivot)^beta).

    D0   : duty cycle at the pivot mass (a probability, 0..1 at the pivot).
    beta : power-law slope; beta > 0 makes massive hosts more likely to be on.
    The clip at 1 enforces "cannot exceed 100% turned on".
    """
    f = D0 * 10.0 ** (beta * (np.asarray(logM, dtype=float) - logM_pivot))
    return np.clip(f, 0.0, 1.0)


def saturation_mass(D0, beta, logM_pivot=11.0):
    """log10 M at which the power-law duty cycle reaches 1 (NaN if it never does)."""
    if beta <= 0 or D0 <= 0:
        return np.nan
    return logM_pivot - np.log10(D0) / beta


# ----------------------------------------------------------------------------
# Catalogue generator (unified: pick a duty-cycle model via the arguments)
# ----------------------------------------------------------------------------
def stochastic_agn_catalog(
    data,
    *,
    # --- luminosity-mass relation ---
    lognormR=22.0, logmnorm=11.0, alphaR=2.0, sigma_logL=0.5,
    # --- duty cycle: choose exactly ONE of these three ---
    duty_cycle=None,                              # scalar -> constant model
    duty_D0=None, duty_beta=None, duty_pivot=11.0,  # power-law model
    p_active=None,                               # explicit per-host probability
    # --- selection / bookkeeping ---
    seed=42, logMcut=11.0, V=1e6, cols=(0, 1, 2, 3),
):
    """Generate one stochastic realisation of the radio-AGN catalogue.

    Returns a dict with 'agn_catalog' = [logM, x, y, z, logL] of active hosts,
    plus 'logL', 'logM', 'p_active' (per eligible host), and 'active_mask'.
    """
    rng = np.random.default_rng(seed)
    mcol, xcol, ycol, zcol = cols

    logM_all = data[:, mcol]
    elig = logM_all > logMcut                    # hard eligibility floor
    logM = logM_all[elig]
    x, y, z = data[elig, xcol], data[elig, ycol], data[elig, zcol]

    # per-host activation probability
    if p_active is not None:
        p = np.clip(np.asarray(p_active, dtype=float)[elig], 0.0, 1.0)
    elif duty_D0 is not None:
        if duty_beta is None:
            raise ValueError("power-law duty cycle needs both duty_D0 and duty_beta")
        p = duty_powerlaw(logM, duty_D0, duty_beta, duty_pivot)
    elif duty_cycle is not None:
        p = duty_constant(logM, duty_cycle)
    else:
        raise ValueError("specify duty_cycle, or duty_D0+duty_beta, or p_active")

    active = rng.random(len(logM)) < p
    logM_a = logM[active]
    x_a, y_a, z_a = x[active], y[active], z[active]

    # power-law mean L(M) with Gaussian scatter in log L
    mu_logL = lognormR + alphaR * (logM_a - logmnorm)
    logL = rng.normal(loc=mu_logL, scale=sigma_logL, size=len(mu_logL))

    agn_catalog = np.column_stack([logM_a, x_a, y_a, z_a, logL])
    return {
        "agn_catalog": agn_catalog,
        "logL": logL,
        "logM": logM_a,
        "mu_logL": mu_logL,
        "p_active": p,            # per ELIGIBLE host, before the Bernoulli draw
        "active_mask": active,
        "n_eligible": int(elig.sum()),
    }


def generate_agn_catalog(data, **kwargs):
    """Backward-compatible deterministic variant (no scatter): sigma_logL = 0."""
    kwargs.setdefault("sigma_logL", 0.0)
    return stochastic_agn_catalog(data, **kwargs)


# ----------------------------------------------------------------------------
# Optional catalogue loader (replaces the old import-time np.loadtxt side effect)
# ----------------------------------------------------------------------------
def load_catalog(path, skiprows=122):
    """Load a text catalogue [log10(mass), x, y, z, ...]. Call explicitly; the
    module no longer reads a file on import (that broke when the file was absent)."""
    return np.loadtxt(path, skiprows=skiprows)
