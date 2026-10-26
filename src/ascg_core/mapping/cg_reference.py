"""Mapped-AA reference data for one task at its approved states (one --run T=DIR each; the benchmark: 400 and 500 K).

Outputs in `out_dir`:
  topology.json        CG beads, types, masses, bonds, angles, 1-2/1-3 exclusions
  mapped_T{T}.npz      bead positions (nm, unwrapped) and boxes for every AA frame
  targets_T{T}.npz     intermolecular partial g(r), bond/angle histograms, per-frame
                       chain sizes and densities, with block-averaged standard errors
  start_T{T}.npz       CG NVT start: the frame closest to the mean volume, scaled to
                       the mean box
  summary.json         scalar observables per T (density, kappa_T, pressure, Rg, Ree)
Bond and angle histograms share one grid per type across both temperatures.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .cg_mapping import (ANGSTROM_PER_NM, AMU_PER_NM3_TO_G_CM3, angle_values, block_se,
                         bond_lengths, build_cg_topology, chain_size, compressibility_per_bar,
                         frame_rdf_counts, intermolecular_pair_counts, map_frame, read_data,
                         read_layout, read_lammps_dump, read_thermo)

BAR_PER_ATM = 1.01325


def _hist(values, edges):
    """Probability density per bin (integrates to 1 over the grid)."""
    h, _ = np.histogram(values, bins=edges)
    total = h.sum()
    return h / (total * np.diff(edges)) if total else h.astype(float)


def whole_plan(bonds, n_atoms):
    """BFS levels [(children, parents)] over the bond graph, one root per connected molecule."""
    from collections import deque
    nbr = [[] for _ in range(n_atoms)]
    for a, b in bonds:
        nbr[a].append(b)
        nbr[b].append(a)
    level, parent = np.full(n_atoms, -1), np.full(n_atoms, -1)
    for root in range(n_atoms):
        if level[root] >= 0:
            continue
        level[root] = 0
        q = deque([root])
        while q:
            u = q.popleft()
            for v in nbr[u]:
                if level[v] < 0:
                    level[v], parent[v] = level[u] + 1, u
                    q.append(v)
    return [(np.where(level == k)[0], parent[level == k]) for k in range(1, level.max() + 1)]


def make_whole(xyz, box, plan):
    """Unwrap every molecule from its root along bonds (minimum image per bond)."""
    x = np.array(xyz, dtype=float)
    for child, par in plan:
        d = x[child] - x[par]
        x[child] = x[par] + d - box * np.round(d / box)
    return x


def read_xtc(path, n_atoms, plan):
    """Yield (time_ps, box_A, whole xyz_A) from a GROMACS .xtc (orthorhombic box)."""
    from MDAnalysis.coordinates.XTC import XTCReader
    with XTCReader(str(path)) as r:
        if r.n_atoms != n_atoms:
            raise ValueError(f'{path}: {r.n_atoms} atoms, expected {n_atoms}')
        for ts in r:
            box = np.asarray(ts.dimensions[:3], float)
            if not np.allclose(ts.dimensions[3:], 90.0):
                raise ValueError('Only orthorhombic boxes are supported')
            yield float(ts.time), box, make_whole(ts.positions, box, plan)


def read_xvg(path):
    """gmx energy .xvg -> {'Step' (time ps), 'Temp' K, 'Press' atm, 'Volume_nm3', 'Density'}."""
    import re
    names, rows = [], []
    for line in Path(path).read_text().splitlines():
        if line.startswith('@'):
            m = re.match(r'@\s+s(\d+)\s+legend\s+"(.+)"', line)
            if m:
                names.append(m.group(2))
        elif line and not line.startswith('#'):
            rows.append([float(x) for x in line.split()])
    a = np.array(rows)
    col = {n.split(' (')[0]: a[:, i + 1] for i, n in enumerate(names)}
    out = {'Step': a[:, 0], 'Temp': col['Temperature'], 'Press': col['Pressure'] / BAR_PER_ATM}
    if 'Volume' in col:
        out['Volume_nm3'] = col['Volume']
    if 'Density' in col:
        out['Density'] = col['Density']
    return out


def run_files(run) -> list[tuple[Path, Path]]:
    """(trajectory, log) pairs of one temperature's production, in time order.

    `run` is a list of (trajectory, log) pairs, a GROMACS protocol branch directory (gmx_protocol: prod.xtc +
    prod_energy.xvg), or a LAMMPS run directory (aa_production.lammpstrj + log.lammps).
    """
    if isinstance(run, (list, tuple)):
        return [(Path(d), Path(l)) for d, l in run]
    run = Path(run)
    if (run/'steps.json').is_file():                    # GROMACS protocol branch (gmx_protocol)
        if not (run/'prod.xtc').is_file() or not (run/'prod_energy.xvg').is_file():
            raise RuntimeError(f'{run}: production trajectory or energy file missing')
        return [(run/'prod.xtc', run/'prod_energy.xvg')]
    return [(run/'aa_production.lammpstrj', run/'log.lammps')]


def _thermo(files):
    """Production thermo rows of all segments, duplicate boundary steps dropped."""
    parts = [read_xvg(log) if str(log).endswith('.xvg') else read_thermo(log) for _, log in files]
    keys = list(parts[0])
    out, seen = {k: [] for k in keys}, set()
    for part in parts:
        for i, step in enumerate(part['Step']):
            if step in seen:
                continue
            seen.add(step)
            for k in keys:
                out[k].append(part[k][i])
    return {k: np.asarray(v) for k, v in out.items()}


def prepare_reference(task, system_dir, runs: dict, out_dir, *, recipes, mappings,
                      r_max_nm=1.5, dr_nm=0.01, bond_bin_nm=0.002, angle_bin_deg=2.0,
                      n_blocks=10):
    """Mapped reference of an AutoPoly build: GROMACS protocol branches (or LAMMPS dumps) per state."""
    system_dir, out = Path(system_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = read_data(system_dir/'system.neutral.data')
    topo = build_cg_topology(task, read_layout(system_dir), data, recipes, mappings)

    mapped = {}
    files = {T: run_files(run) for T, run in runs.items()}
    plan = None                                  # bond-graph unwrapping, for GROMACS trajectories
    for T in runs:
        pos, box, steps = [], [], []
        for dump, _ in files[T]:
            if str(dump).endswith('.xtc'):
                if plan is None:
                    plan = whole_plan(data['bonds'], len(data['mass']))
                frames = read_xtc(dump, len(data['mass']), plan)
            else:
                frames = read_lammps_dump(dump, len(data['mass']))
            for step, b, xyz in frames:
                if steps and step <= steps[-1]:
                    continue                 # segment boundary frame written twice
                pos.append(map_frame(xyz, data['mass'], topo.atom_bead, topo.bead_masses_amu))
                box.append(b / ANGSTROM_PER_NM)
                steps.append(step)
        mapped[T] = (np.array(pos), np.array(box), np.array(steps))
    return write_reference(task, topo, mapped, {T: _thermo(files[T]) for T in runs}, out, r_max_nm=r_max_nm,
                           dr_nm=dr_nm, bond_bin_nm=bond_bin_nm, angle_bin_deg=angle_bin_deg, n_blocks=n_blocks)


def write_reference(task, topo, mapped, thermo, out, *, r_max_nm=1.5, dr_nm=0.01, bond_bin_nm=0.002,
                    angle_bin_deg=2.0, n_blocks=10, source=None):
    """The mapped-reference files from a CG topology and mapped frames, for any AA source.
    mapped: {state: (bead positions nm [F, N, 3] unwrapped per molecule, boxes nm [F, 3], steps or times [F])};
    thermo: {state: {'Temp' K, 'Press' atm, optional 'Volume_nm3'} or None (no energy file: pressure and temperature
    are not reported; the compressibility then comes from the frame volumes of a fluctuating box)}.
    Bond and angle histograms share one grid per type across the states."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out/'topology.json').write_text(json.dumps(topo.to_dict()) + '\n')
    for T, (P, B, steps) in mapped.items():
        np.savez_compressed(out/f'mapped_T{T}.npz', pos_nm=P.astype(np.float32), box_nm=B.astype(np.float32),
                            step=np.asarray(steps))

    bond_types = sorted({topo.bond_type(b) for b in topo.bonds})
    angle_types = sorted({topo.angle_type(a) for a in topo.angles})
    bond_sel = {k: [i for i, b in enumerate(topo.bonds) if topo.bond_type(b) == k] for k in bond_types}
    angle_sel = {k: [i for i, a in enumerate(topo.angles) if topo.angle_type(a) == k] for k in angle_types}
    all_bonds = {k: np.concatenate([bond_lengths(p, topo.bonds)[bond_sel[k]]
                                    for P, _, _ in mapped.values() for p in P]) for k in bond_types}
    bond_edges = {k: np.arange(max(0.0, v.min() - 0.02), v.max() + 0.02 + bond_bin_nm, bond_bin_nm)
                  for k, v in all_bonds.items()}
    angle_edges = np.deg2rad(np.arange(0.0, 180.0 + angle_bin_deg / 2, angle_bin_deg))
    r_edges = np.arange(0.0, r_max_nm + dr_nm / 2, dr_nm)
    shell = 4 / 3 * np.pi * (r_edges[1:]**3 - r_edges[:-1]**3)
    n_pairs = intermolecular_pair_counts(topo.bead_types, topo.bead_molecule, topo.type_names)
    total_mass = float(topo.bead_masses_amu.sum())

    summary = {'task': task, 'n_beads': len(topo.bead_types), 'type_names': topo.type_names,
               'bond_types': bond_types, 'angle_types': angle_types, 'states': {}}
    if source:
        summary['source'] = source
    for T, (P, B, _) in mapped.items():
        if r_max_nm >= B.min() / 2:
            raise ValueError(f'r_max {r_max_nm} nm >= half the smallest box {B.min()/2:.3f} nm at {T} K')
        vol = np.prod(B, axis=1)
        g_frames = {k: [] for k in n_pairs}
        bond_frames = {k: [] for k in bond_types}
        angle_frames = {k: [] for k in angle_types}
        rg_f, ree_f = [], []
        for p, b, v in zip(P, B, vol):
            counts = frame_rdf_counts(p, b, topo.bead_types, topo.bead_molecule, topo.type_names, r_edges)
            for k, c in counts.items():
                g_frames[k].append(c / (n_pairs[k] * shell / v) if n_pairs[k] else np.zeros_like(shell))
            bl, av = bond_lengths(p, topo.bonds), angle_values(p, topo.angles)
            for k in bond_types:
                bond_frames[k].append(_hist(bl[bond_sel[k]], bond_edges[k]))
            for k in angle_types:
                angle_frames[k].append(_hist(av[angle_sel[k]], angle_edges))
            rg, ree = chain_size(p, topo.bead_masses_amu, topo.bead_molecule)
            rg_f.append(rg.mean())
            ree_f.append(np.sqrt((ree**2).mean()))
        rho = total_mass * AMU_PER_NM3_TO_G_CM3 / vol
        arrays = {'r_nm': 0.5 * (r_edges[1:] + r_edges[:-1]), 'angle_rad': 0.5 * (angle_edges[1:] + angle_edges[:-1]),
                  'density_g_cm3': rho, 'volume_nm3': vol, 'rg_nm': np.array(rg_f), 'ree_rms_nm': np.array(ree_f)}
        for k, fr in g_frames.items():
            fr = np.array(fr)
            arrays[f'rdf/{k}'], arrays[f'rdf_se/{k}'] = fr.mean(0), block_se(fr, n_blocks)
        for k in bond_types:
            fr = np.array(bond_frames[k])
            arrays[f'bond_edges/{k}'] = bond_edges[k]
            arrays[f'bond/{k}'], arrays[f'bond_se/{k}'] = fr.mean(0), block_se(fr, n_blocks)
        for k in angle_types:
            fr = np.array(angle_frames[k])
            arrays[f'angle/{k}'], arrays[f'angle_se/{k}'] = fr.mean(0), block_se(fr, n_blocks)
        np.savez_compressed(out/f'targets_T{T}.npz', **arrays)

        i = int(np.argmin(np.abs(vol - vol.mean())))
        mean_box = B.mean(0)
        np.savez_compressed(out/f'start_T{T}.npz', pos_nm=(P[i] * mean_box / B[i]).astype(np.float64),
                            box_nm=mean_box, source_frame=i)

        from ..ibi.states import parse_one
        sp = parse_one(T)
        th = thermo.get(T)
        npt = float(vol.std() / vol.mean()) > 1e-6                  # a fluctuating box: an NPT trajectory
        if th is not None:
            press_bar = th['Press'] * BAR_PER_ATM
            kappa = compressibility_per_bar(th.get('Volume_nm3', vol), float(T))
            p_mean, p_se, t_mean = float(press_bar.mean()), float(block_se(press_bar, n_blocks)), float(th['Temp'].mean())
        else:
            kappa = compressibility_per_bar(vol, float(sp.T)) if npt else None
            p_mean = p_se = t_mean = None
        summary['states'][str(T)] = {
            'temperature_K': sp.T, 'pressure_atm': sp.P, 'pressure_bar_target': round(sp.pressure_bar, 6),
            'frames': len(P), 'box_nm_mean': mean_box.tolist(),
            'density_g_cm3': float(rho.mean()), 'density_se': float(block_se(rho, n_blocks)),
            # stationarity: last fifth minus first fifth of the production run
            'density_drift_g_cm3': float(rho[-max(1, len(rho)//5):].mean() - rho[:max(1, len(rho)//5)].mean()),
            # thermo volumes (every 1 ps in the GROMACS protocol) when available, else frame volumes
            'kappa_T_per_bar': kappa,
            'pressure_bar_mean': p_mean, 'pressure_bar_se': p_se,
            'temperature_K_mean': t_mean,
            'rg_nm': float(np.mean(rg_f)), 'rg_se': float(block_se(rg_f, n_blocks)),
            'ree_rms_nm': float(np.mean(ree_f)), 'ree_se': float(block_se(ree_f, n_blocks)),
            'r_max_over_half_box': float(r_max_nm / (B.min() / 2)),
            **({} if th is not None else {'ensemble': 'NPT' if npt else 'NVT', 'thermo': None}),
        }
    (out/'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary
