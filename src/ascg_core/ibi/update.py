"""Multistate IBI updates of one shared potential set.

Pair a-b, iteration n -> n+1, over the fit's approved states s (ibi.states; N >= 1):

    U^(n+1)(r) = U^(n)(r) + lambda sum_s w_s k_B T_s clip( ln[ g_CG,s(r) / g_AA,s(r) ], -c, c ),  sum_s w_s = 1

with the plan's defaults lambda = 0.2, w = (0.5, 0.5), c = 2. Bonded terms use the same rule with the bond-length
or angle probability densities on the reference grids (the Jacobians r^2 / sin(theta) cancel in the ratio).

Support: the log ratio is used only where both the target and the CG distribution are physically sampled at every
state (P > floor x max P); bins outside are never read as a perfect fit. The correction (defined on the observable
grid) is smoothed with a Gaussian of a few bins, held at its value at the edge of the support (so walls are shifted
continuously, not bent), interpolated onto the table grid with a cubic spline and, for pairs, multiplied by the
tables' C2 switch so U and F stay zero at r_c. Pressure (density) correction, separately tagged (Reith et al.,
Macromolecules 2003): A (1 - r/r_c) S(r) added to every pair table with A = sum_s w_s A_s,
A_s = -0.1 k_B T_s sign(dP_s) min(1, 3e-4 |dP_s| / bar), dP_s = P_CG,s - P_target.

T-dependent model U(x, T) = A(x) + B(x) T: every state gets its own correction
dU_s = lambda k_B T_s clip(ln[P_CG,s / P_AA,s], -c, c) (own support, smoothing and edge hold) and its own Reith ramp
A_s (1 - r/r_c) S(r); dA, dB solve dA + dB T_s = dU_s (one state: dB = 0; two: exact; more: least squares weighted
by the state weights).

Virial pressure correction (protocol v1; the Reith step shrinks with |dP|):
the same ramp shape, with the amplitude from the first-order (fixed-structure) virial response of the previous
run's nonbonded pair-distance histograms h_k(r) (analysis.pair_histograms; all pairs the tables act on):
dP[dU] = -1 / (3 V) sum_k sum_b h_k(r_b) r_b dU_k'(r_b). A_s = scale (P_target - P_s - dP_s[IBI step]) / dP_s[ramp of
unit amplitude], clipped to +-cap k_B T_s: the ramp cancels the measured deviation and the predicted pressure
change of the same iteration's pair correction (in the spirit of the Wang-Junghans-Kremer correction in VOTCA).
"""
from __future__ import annotations

import numpy as np

from .base_tables import _c2_switch, _smooth

DEFAULTS = {'lam': 0.2, 'cap': 2.0, 'floor_rel': 1e-3, 'smooth_bins': 1.0}     # state weights: ibi.states.weights


def log_ratio_correction(x, cg_by_T, aa_by_T, kts, weights, *, lam, cap, floor_rel, smooth_bins):
    """lambda sum_s w_s kT_s clip(ln(P_CG,s / P_AA,s), -c, c) on the observable grid x, and the support mask.
    cg_by_T / aa_by_T: {T: distribution on x}; kts: {T: k_B T}; weights: {T: w}, summing to 1."""
    temps = list(weights)
    if abs(sum(weights.values()) - 1.0) > 1e-9:
        raise ValueError('state weights must sum to 1')
    support = np.ones(len(x), bool)
    for T in temps:
        aa, cg = np.asarray(aa_by_T[T], float), np.asarray(cg_by_T[T], float)
        support &= (aa > floor_rel * aa.max()) & (cg > floor_rel * aa.max())
    corr = np.zeros(len(x))
    if not support.any():
        return corr, support
    for T in temps:
        aa, cg = np.asarray(aa_by_T[T], float), np.asarray(cg_by_T[T], float)
        with np.errstate(divide='ignore', invalid='ignore'):
            lr = np.where(support, np.log(cg / aa), 0.0)
        corr += weights[T] * kts[T] * np.clip(lr, -cap, cap)
    corr *= lam
    # smooth inside the support; hold the edge values outside it
    idx = np.flatnonzero(support)
    inner = np.interp(np.arange(len(x)), idx, corr[idx])
    inner = _smooth(inner, smooth_bins)
    corr = np.where(np.arange(len(x)) < idx[0], inner[idx[0]],
                    np.where(np.arange(len(x)) > idx[-1], inner[idx[-1]], inner))
    return corr, support


def apply_correction(pot, x_obs, corr, *, r_switch=None, rc=None):
    """New Potential: pot + corr interpolated onto the table grid (held constant beyond x_obs); pair tables are
    multiplied by the C2 switch between r_switch and rc."""
    from scipy.interpolate import CubicSpline
    xs = np.clip(pot.x, x_obs[0], x_obs[-1])
    d = CubicSpline(x_obs, corr)(xs)
    if pot.kind == 'pair':
        s, _ = _c2_switch(pot.x, r_switch, rc)
        d = d * s
    return pot.copy(u=pot.u + d)


def _target(target_bar, T):
    """The state's pressure target (bar): one number for every state, or {state: bar}."""
    return float(target_bar[T]) if isinstance(target_bar, dict) else float(target_bar)


def reith_amplitude(pressure_bar_by_T, kts, weights, target_bar=1.01325):
    """Shared pressure-correction amplitude A (kJ/mol) from the states' pressures and targets (bar)."""
    a = 0.0
    for T, w in weights.items():
        dp = pressure_bar_by_T[T] - _target(target_bar, T)
        a += w * (-0.1 * kts[T] * np.sign(dp) * min(1.0, 3e-4 * abs(dp)))
    return float(a)


def pressure_ramp(pot, amplitude, *, r_switch, rc):
    """pot + A (1 - r/rc) S(r) (pair tables)."""
    s, _ = _c2_switch(pot.x, r_switch, rc)
    return pot.copy(u=pot.u + amplitude * np.clip(1 - pot.x / rc, 0.0, None) * s)


def state_corrections(x, cg_by_T, aa_by_T, kts, *, lam, cap, floor_rel, smooth_bins):
    """Per-state IBI corrections {T: dU_T} on the observable grid (T-dependent model)."""
    return {T: log_ratio_correction(x, {T: cg_by_T[T]}, {T: aa_by_T[T]}, {T: kts[T]}, {T: 1.0}, lam=lam, cap=cap,
                                    floor_rel=floor_rel, smooth_bins=smooth_bins)[0] for T in cg_by_T}


def apply_tcorrection(tpot, x_obs, corr_by_T, *, r_switch=None, rc=None, weights=None):
    """New TPotential: per-state corrections (on x_obs, held constant beyond it; pair: times the C2 switch) mapped
    onto A and B."""
    from scipy.interpolate import CubicSpline
    from .tables import TPotential, solve_ab
    xs = np.clip(tpot.x, x_obs[0], x_obs[-1])
    sw = _c2_switch(tpot.x, r_switch, rc)[0] if tpot.kind == 'pair' else 1.0
    da, db = solve_ab({T: CubicSpline(x_obs, c)(xs) * sw for T, c in corr_by_T.items()}, weights)
    return TPotential(tpot.kind, tpot.x.copy(), tpot.a + da, tpot.b + db)


def reith_amplitudes(pressure_bar_by_T, kts, target_bar=1.01325):
    """Per-state Reith amplitudes {T: A_T} (kJ/mol) towards each state's target pressure (bar)."""
    return {T: float(-0.1 * kts[T] * np.sign(p - _target(target_bar, T)) * min(1.0, 3e-4 * abs(p - _target(target_bar, T))))
            for T, p in pressure_bar_by_T.items()}


def tpressure_ramp(tpot, amp_by_T, *, r_switch, rc, weights=None):
    """TPotential + per-state ramps A_T (1 - r/rc) S(r), mapped onto A and B."""
    from .tables import TPotential, solve_ab
    shape = np.clip(1 - tpot.x / rc, 0.0, None) * _c2_switch(tpot.x, r_switch, rc)[0]
    da, db = solve_ab({T: amp * shape for T, amp in amp_by_T.items()}, weights)
    return TPotential(tpot.kind, tpot.x.copy(), tpot.a + da, tpot.b + db)



def ramp_shape(x, *, r_switch, rc):
    """(1 - r/rc) S(r) and its derivative."""
    s, ds = _c2_switch(np.asarray(x, float), r_switch, rc)
    lin = np.clip(1 - np.asarray(x, float) / rc, 0.0, None)
    return lin * s, np.where(np.asarray(x) < rc, -1.0 / rc, 0.0) * s + lin * ds


def virial_pressure_change(hist, edges, volume_nm3, dudr):
    """First-order pressure change (bar) at fixed structure of pair-table changes; dudr: {key: callable r -> dU'}."""
    from .fastmd_io import BAR_PER_KJ_MOL_NM3
    r = 0.5 * (np.asarray(edges)[1:] + np.asarray(edges)[:-1])
    w = sum(float(np.sum(np.asarray(hist[k]) * r * dudr[k](r))) for k in dudr)
    return -w / (3.0 * float(volume_nm3)) * BAR_PER_KJ_MOL_NM3


def table_change_derivative(old, new, T=None):
    """Callable r -> d/dr [U_new - U_old] (pair tables on a common grid; TPotential at T or Potential)."""
    if T is None:
        du = new.u - old.u
    else:
        du = (new.a - old.a) + (new.b - old.b) * float(T)
    d = np.gradient(du, old.x)
    return lambda r: np.interp(r, old.x, d, left=0.0, right=0.0)


def virial_amplitudes(p_by_T, dp_step_by_T, hist_by_T, edges, vol_by_T, kts, *, r_switch, rc, scale=0.5, cap_kt=0.3,
                      target_bar=1.01325):
    """Per-state ramp amplitudes {T: A_T} (kJ/mol) and the predicted responses {T: bar per kJ/mol}."""
    amps, resp = {}, {}
    for T, p in p_by_T.items():
        ds = lambda r: ramp_shape(r, r_switch=r_switch, rc=rc)[1]
        d = virial_pressure_change(hist_by_T[T], edges, vol_by_T[T], {k: ds for k in hist_by_T[T]})
        resp[T] = d
        a = scale * (_target(target_bar, T) - p - dp_step_by_T.get(T, 0.0)) / d if d > 0 else 0.0
        amps[T] = float(np.clip(a, -cap_kt * kts[T], cap_kt * kts[T]))
    return amps, resp
