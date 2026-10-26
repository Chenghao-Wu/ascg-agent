"""Fixed Boltzmann-inverted CG tables (pair, bond, angle) shared by 400 and 500 K.

Each table is the average of the two temperatures' potentials of mean force,
W_T(x) = -k_B T ln[P_T(x) / J(x)] with J = r^2 (bonds), sin(theta) (angles) or 1
(intermolecular g(r)), as decided for the study (no explicit T dependence).
Regularization:
  * the PMFs are smoothed (Gaussian kernel, a few bins) where both T are sampled;
  * pair: below the first r where g > g_floor at both T, a repulsive
    A (r_c/r)^12 + B wall continues the table with matched value and slope;
    the tail is switched to zero between r_switch and r_cut with a C2 polynomial;
  * bonds/angles: outside the sampled range a quadratic wall continues the
    table with matched value and slope and curvature >= the local fit.
Tables are returned on a uniform grid with values and analytic slopes for C1
cubic Hermite interpolation (`hermite`), in kJ/mol and nm / rad.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

KB_KJ_PER_MOL_K = 0.00831446261815324


@dataclass
class Table:
    x0: float
    dx: float
    value: np.ndarray        # kJ/mol on x0 + i*dx
    slope: np.ndarray        # kJ/mol per unit x

    def __call__(self, x):
        return hermite(np.asarray(x, float), self.x0, self.dx, self.value, self.slope)

    def to_dict(self):
        return {'x0': self.x0, 'dx': self.dx, 'value': self.value.tolist(), 'slope': self.slope.tolist()}


def hermite(x, x0, dx, value, slope, xp=np):
    """C1 cubic Hermite interpolation on a uniform grid; clamps outside the grid."""
    n = value.shape[0]
    t = xp.clip((x - x0) / dx, 0.0, n - 1 - 1e-9)
    i = xp.floor(t).astype(int) if xp is np else xp.floor(t).astype('int32')
    u = t - i
    h00, h10 = 2*u**3 - 3*u**2 + 1, u**3 - 2*u**2 + u
    h01, h11 = -2*u**3 + 3*u**2, u**3 - u**2
    return h00*value[i] + h10*dx*slope[i] + h01*value[i+1] + h11*dx*slope[i+1]


def _smooth(y, sigma_bins):
    if sigma_bins <= 0:
        return y
    k = np.arange(-int(4*sigma_bins), int(4*sigma_bins) + 1)
    w = np.exp(-0.5*(k/sigma_bins)**2)
    w /= w.sum()
    pad = np.pad(y, len(k)//2, mode='edge')
    return np.convolve(pad, w, mode='valid')


def _pmf(x, prob, jac, T, floor):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(prob > floor, -KB_KJ_PER_MOL_K*T*np.log(prob/jac), np.nan)


def _c2_switch(r, r_on, r_off):
    """1 below r_on, 0 above r_off, C2 quintic in between; returns (S, dS/dr)."""
    t = np.clip((r - r_on)/(r_off - r_on), 0, 1)
    s = 1 - 10*t**3 + 15*t**4 - 6*t**5
    ds = (-30*t**2 + 60*t**3 - 30*t**4)/(r_off - r_on)
    return s, ds


def with_pressure_ramp(table: Table, A, *, r_switch=1.3, r_cut=1.5):
    """Table + A (1 - r/r_cut) S(r): the IBI pressure correction (Reith et al. 2003), switched with
    the table's own C2 switch so forces stay continuous at r_cut. A < 0 (attractive) lowers the
    pressure; acting where g(r) ~ 1, it shifts the virial with little change in structure."""
    x = table.x0 + table.dx * np.arange(len(table.value))
    s, ds = _c2_switch(x, r_switch, r_cut)
    lin = np.clip(1 - x / r_cut, 0.0, None)
    return Table(table.x0, table.dx, table.value + A * lin * s,
                 table.slope + A * (np.where(x < r_cut, -1 / r_cut, 0.0) * s + lin * ds))


def pair_table(r, g_by_T: dict, *, r_switch=1.3, r_cut=1.5, g_floor=1e-3, smooth_bins=1.5, dx=0.001):
    """BI pair table from intermolecular g(r) at both temperatures (r in nm, bin centres)."""
    r = np.asarray(r, float)
    W = [_pmf(r, np.asarray(g, float), 1.0, float(T), g_floor) for T, g in g_by_T.items()]
    ok = np.all([np.isfinite(w) for w in W], axis=0)
    if not ok.any():
        raise ValueError('No r with g above the floor at both temperatures')
    i0 = int(np.argmax(ok))
    sel = np.arange(i0, len(r))
    sel = sel[ok[sel]]
    from scipy.interpolate import CubicSpline
    w = np.mean([_smooth(np.interp(r[sel], r[sel], wi[sel]), smooth_bins) for wi in W], axis=0)
    grid = np.arange(0.0, r_cut + dx/2, dx)
    cs = CubicSpline(r[sel], w)              # smooth values with consistent slopes
    inside = (grid >= r[sel][0]) & (grid <= r[sel][-1])
    v = np.where(inside, cs(np.clip(grid, r[sel][0], r[sel][-1])), 0.0)
    dv = np.where(inside, cs(np.clip(grid, r[sel][0], r[sel][-1]), 1), 0.0)
    v[grid > r[sel][-1]] = w[-1]
    # repulsive wall below the first sampled r: A (rc/r)^12 + B with matched value/slope
    rc = r[sel][0]
    vc = np.interp(rc, grid, v)
    sc = min(np.interp(rc, grid, dv), -1.0)        # keep the wall repulsive
    A = -sc*rc/12
    inner = grid < rc
    rr = np.maximum(grid[inner], 0.05*rc)
    v[inner] = A*(rc/rr)**12 - A + vc
    dv[inner] = -12*A/rr*(rc/rr)**12
    s, ds = _c2_switch(grid, r_switch, r_cut)
    return Table(0.0, dx, v*s, dv*s + v*ds)


def bonded_table(x, p_by_T: dict, *, kind: str, floor_rel=1e-3, smooth_bins=1.0, dx=None, x_range=None):
    """BI bond (x = r nm) or angle (x = theta rad) table averaged over the temperatures."""
    x = np.asarray(x, float)
    jac = x**2 if kind == 'bond' else np.sin(np.clip(x, 1e-6, np.pi - 1e-6))
    W = []
    for T, p in p_by_T.items():
        p = np.asarray(p, float)
        W.append(_pmf(x, p, jac, float(T), floor_rel*p.max()))
    ok = np.all([np.isfinite(w) for w in W], axis=0)
    idx = np.where(ok)[0]
    if len(idx) < 5:
        raise ValueError(f'Too few sampled {kind} bins')
    lo, hi = idx[0], idx[-1]
    seg = np.arange(lo, hi + 1)
    w = np.mean([_smooth(np.interp(x[seg], x[idx], wi[idx]), smooth_bins) for wi in W], axis=0)
    w -= w.min()
    if dx is None:
        dx = (x[1] - x[0])/4
    if x_range is None:
        x_range = (0.0, x[-1] + (x[1]-x[0])) if kind == 'bond' else (0.0, np.pi)
    from scipy.interpolate import CubicSpline
    grid = np.arange(x_range[0], x_range[1] + dx/2, dx)
    cs = CubicSpline(x[seg], w)              # smooth values with consistent slopes
    xc = np.clip(grid, x[seg][0], x[seg][-1])
    v, dv = cs(xc), cs(xc, 1)
    # quadratic walls outside [x_lo, x_hi], matched value and slope, curvature from the edge fit
    for edge, side in ((x[seg][0], -1), (x[seg][-1], +1)):
        n = min(5, len(seg))
        xs = x[seg][:n] if side < 0 else x[seg][-n:]
        ws = w[:n] if side < 0 else w[-n:]
        c = max(np.polyfit(xs, ws, 2)[0], 1.0) if n >= 3 else 1.0
        we, se = float(cs(edge)), float(cs(edge, 1))
        if side < 0:
            se = min(se, 0.0)
        else:
            se = max(se, 0.0)
        out = grid < edge if side < 0 else grid > edge
        d = grid[out] - edge
        v[out] = we + se*d + c*d**2
        dv[out] = se + 2*c*d
    return Table(float(grid[0]), float(dx), v, dv)
