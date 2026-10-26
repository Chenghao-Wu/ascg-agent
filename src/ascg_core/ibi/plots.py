"""Report figure of one IBI task (report/structure_density.png).

    python -m ascg_core.ibi.plots --ref runs/cg_ref/pe --task-dir runs/ibi/pe

One row per state of the fit: mapped-AA targets (dashed) vs the held-out CG runs (solid, mean over seeds): partial
RDFs, bond-length and angle (and dihedral) distributions, NRMSE in the legends. Last row: IBI convergence (worst pair
RDF NRMSE, NVT pressure, NPT density error with the 5 % gate) per state, and the held-out densities.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..plotting import GRID, INK, INK2, MUTED, SERIES, TYPE_COLORS, style as _style
from .analysis import Grids, nrmse
from .states import describe, of_run, parse_one
from .system import load_system


_pretty = lambda T: parse_one(T).pretty()         # '400 K' / '400 K, 1000 atm'
def state_colors(temps):
    """{T: colour}: the validated blue / orange pair (cold, hot) for one or two states, else a sequential ramp in
    temperature order (matplotlib plasma, 0.1-0.8: perceptually ordered, cold dark to hot light)."""
    temps = sorted(temps)
    if len(temps) <= 2:
        return dict(zip(temps, SERIES))
    from matplotlib import colormaps
    cmap = colormaps['plasma']
    return {T: cmap(0.1 + 0.7 * i / (len(temps) - 1)) for i, T in enumerate(temps)}


T_COLORS = state_colors((400, 500))              # the benchmark pair (older figures)


def task_figure(ref_dir, task_dir, out_png):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _style(plt)
    ref, task = Path(ref_dir), Path(task_dir)
    system = load_system(ref)
    final = task/'cg'/'ibi'/'final'
    val = json.loads((final/'validation.json').read_text())
    state = json.loads((task/'workflow'/'state.json').read_text())
    from .dihedrals import style_of
    with_dih = bool(style_of(state['settings'].get('dihedrals')))
    if with_dih:
        system.add_dihedrals(style_of(state['settings'].get('dihedrals')))
    temps = of_run(state)
    colors = state_colors(temps)
    nrow = len(temps) + 1
    fig, ax = plt.subplots(nrow, 4 if with_dih else 3, figsize=(14.5 if with_dih else 11, 2.75 * nrow),
                           squeeze=False)
    last = nrow - 1
    if with_dih:
        ax[last, 3].axis('off')
    for row, T in enumerate(temps):
        t = dict(np.load(ref/f'targets_T{T}.npz'))
        if with_dih:
            t.update(dict(np.load(task/'targets'/f'dihedrals_T{T}.npz')))
        g = Grids(t, system)
        seeds = sorted((final/f'holdout_T{T}').glob('seed_*/distributions.npz'))
        cg = [np.load(s) for s in seeds]
        mean = lambda key: np.mean([c[key] for c in cg], axis=0)
        for j, k in enumerate(system.pair_keys):
            c = TYPE_COLORS[j % len(TYPE_COLORS)]
            y = mean(f'rdf/{k}')
            ax[row, 0].plot(g.r, t[f'rdf/{k}'], color=c, lw=1.0, ls=(0, (4, 2)))
            ax[row, 0].plot(g.r, y, color=c, lw=1.4, label=f'{k}  {nrmse(y, t[f"rdf/{k}"]):.3f}')
        for j, k in enumerate(system.bond_types):
            c = TYPE_COLORS[j % len(TYPE_COLORS)]
            x = 0.5 * (g.bond_edges[k][1:] + g.bond_edges[k][:-1])
            y = mean(f'bond/{k}')
            ax[row, 1].plot(x, t[f'bond/{k}'], color=c, lw=1.0, ls=(0, (4, 2)))
            ax[row, 1].plot(x, y, color=c, lw=1.4, label=f'{k}  {nrmse(y, t[f"bond/{k}"]):.3f}')
        for j, k in enumerate(system.angle_types):
            c = TYPE_COLORS[j % len(TYPE_COLORS)]
            y = mean(f'angle/{k}')
            ax[row, 2].plot(np.degrees(g.theta), t[f'angle/{k}'], color=c, lw=1.0, ls=(0, (4, 2)))
            ax[row, 2].plot(np.degrees(g.theta), y, color=c, lw=1.4, label=f'{k}  {nrmse(y, t[f"angle/{k}"]):.3f}')
        s = val['states'][str(T)]
        ax[row, 0].set(xlim=(0.3, 1.5), xlabel='r (nm)', ylabel='g(r)')
        ax[row, 0].set_ylim(0, 1.2 * ax[row, 0].get_ylim()[1])           # headroom for the density note
        ax[row, 0].set_title(f'{_pretty(T)}  intermolecular RDF', loc='left', fontsize=9)
        ax[row, 0].text(0.98, 0.97, f'held-out NPT density\n{100 * s["density_error_mean"]:+.1f} % vs AA '
                        f'({val["seeds"]} seeds)', transform=ax[row, 0].transAxes, ha='right', va='top',
                        fontsize=7, color=INK2)
        if len(system.bond_types):
            lo = min(x[t[f'bond/{k}'] > 1e-3 * t[f'bond/{k}'].max()].min()
                     for k in system.bond_types for x in [0.5 * (g.bond_edges[k][1:] + g.bond_edges[k][:-1])])
            hi = max(x[t[f'bond/{k}'] > 1e-3 * t[f'bond/{k}'].max()].max()
                     for k in system.bond_types for x in [0.5 * (g.bond_edges[k][1:] + g.bond_edges[k][:-1])])
            ax[row, 1].set(xlim=(lo - 0.01, hi + 0.01), xlabel='bond length (nm)', ylabel='P(b) (1/nm)')
        else:
            ax[row, 1].text(0.5, 0.5, 'no bonds in this mapping', transform=ax[row, 1].transAxes, ha='center',
                            color=INK2)
        ax[row, 1].set_title(f'{_pretty(T)}  bond length', loc='left', fontsize=9)
        if len(system.angle_types):
            th_on = [np.degrees(g.theta)[t[f'angle/{k}'] > 1e-3 * t[f'angle/{k}'].max()] for k in system.angle_types]
            ax[row, 2].set(xlim=(min(x.min() for x in th_on) - 5, 180), xlabel='angle (deg)',
                           ylabel='P(theta) (1/rad)')
        else:
            ax[row, 2].text(0.5, 0.5, 'no angles in this mapping', transform=ax[row, 2].transAxes, ha='center',
                            color=INK2)
        ax[row, 2].set_title(f'{_pretty(T)}  bond angle', loc='left', fontsize=9)
        if with_dih:
            for j, k in enumerate(system.dihedral_types):
                c = TYPE_COLORS[j % len(TYPE_COLORS)]
                y = mean(f'dihedral/{k}')
                ax[row, 3].plot(np.degrees(t['phi']), t[f'dihedral/{k}'], color=c, lw=1.0, ls=(0, (4, 2)))
                ax[row, 3].plot(np.degrees(t['phi']), y, color=c, lw=1.4,
                                label=f'{k}  {nrmse(y, t[f"dihedral/{k}"]):.3f}')
            ax[row, 3].set(xlim=(-180, 180), xlabel='dihedral (deg)', ylabel='P(phi) (1/rad)')
            ax[row, 3].set_title(f'{_pretty(T)}  dihedral ({system.dihedral_style})', loc='left', fontsize=9)
        for j, a in enumerate(ax[row]):
            a.grid(color=GRID, lw=0.6)
            if a.get_legend_handles_labels()[0]:            # empty for a family the mapping does not have
                a.legend(frameon=False, fontsize=7, title='NRMSE (solid CG, dashed AA)', title_fontsize=7,
                         loc='lower right' if j == 0 else 'best')   # RDF: clear of the density note
    h = state['history']
    it = [e['iteration'] for e in h]
    for T in temps:
        c = colors[T]
        ax[last, 0].plot(it, [max(e[f'rdf_{T}'].values()) for e in h], color=c, lw=1.4, label=f'{_pretty(T)}')
        ax[last, 1].plot(it, [e[f'p_bar_{T}'] for e in h], color=c, lw=1.4, label=f'{_pretty(T)}')
        pts = [(e['iteration'], 100 * e[f'density_error_{T}']) for e in h if f'density_error_{T}' in e]
        if pts:
            ax[last, 2].plot(*zip(*pts), color=c, lw=1.2, marker='o', ms=4, label=f'{_pretty(T)} (NPT check)')
        ax[last, 2].plot([it[-1] + 1.5], [100 * val['states'][str(T)]['density_error_mean']], color=c, marker='D',
                         ms=6, ls='none')
    ax[last, 0].set(yscale='log', xlabel='IBI iteration', ylabel='worst pair RDF NRMSE')
    ax[last, 0].axhline(0.15, color=MUTED, lw=0.8, ls=':')
    ax[last, 1].set(yscale='symlog', xlabel='IBI iteration', ylabel='NVT pressure (bar)')
    if all(T.at_1atm for T in temps):
        ax[last, 1].axhline(1.0, color=INK2, lw=0.8)
    else:                                               # each state's own pressure target
        for T in temps:
            ax[last, 1].axhline(T.pressure_bar, color=colors[T], lw=0.8, ls=(0, (4, 2)))
    ax[last, 2].axhspan(-5, 5, color=GRID, alpha=0.6, lw=0)
    ax[last, 2].set(xlabel='IBI iteration', ylabel='density error vs AA (%)', ylim=(-10, 10))
    ax[last, 2].text(it[-1] + 1.5, 7.5, 'held-out', color=INK2, fontsize=7, ha='center')
    ramp_on = [e['iteration'] for e in h if e.get('pressure_ramp_kj')]
    methods = {inv['settings'].get('pressure_correction') for inv in state.get('invocations', [])} - {None, 'none'}
    method = '/'.join(sorted(methods)) if methods else 'Reith'          # runs before invocations were recorded
    p_title = (f'NVT pressure ({method} ramp from iteration {ramp_on[0]})' if ramp_on else 'NVT pressure (no ramp)')
    for a, title in zip(ax[last][:3], ('RDF convergence (dotted: 0.15 gate)', p_title,
                                'NPT density (band: 5 % gate; clipped at +-10 %)')):
        a.grid(color=GRID, lw=0.6)
        if a.get_legend_handles_labels()[0]:                # no NPT checks: nothing to label
            a.legend(frameon=False, fontsize=7)
        a.set_title(title, loc='left', fontsize=9)
    fam = [e['family'] for e in h]
    tl = describe(temps)
    model = ('one table set at ' + tl if len(temps) == 1 else
             f'U(x, T) = A(x) + B(x) T over {tl}' if state['settings'].get('model', 'shared') == 'tdep'
             else f'one shared potential for {tl}')
    dih_txt = f', {fam.count("dihedral")} dihedral' if with_dih else ''
    dih_model = f', {system.dihedral_style} dihedrals' if with_dih else ''
    fig.suptitle(f'{system.task.upper()}  {"multistate " if len(temps) > 1 else ""}IBI (fastMD), {model}{dih_model}; '
                 f'{len(it) - 1} iterations ({fam.count("pair")} pair, {fam.count("bond")} bond, {fam.count("angle")} '
                 f'angle{dih_txt}); {"VALIDATED" if val["validated"] else "NOT VALIDATED"}', x=0.01, ha='left',
                 fontsize=10, color=INK)
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref', required=True)
    ap.add_argument('--task-dir', required=True)
    ap.add_argument('--out', default=None, help='default: <task-dir>/report/structure_density.png')
    a = ap.parse_args(argv)
    out = a.out or str(Path(a.task_dir)/'report'/'structure_density.png')
    task_figure(a.ref, a.task_dir, out)
    print(out)


if __name__ == '__main__':
    main()
