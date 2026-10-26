"""Synthetic AA test system for the bring-your-own tests: N copies of a molecule (RDKit 3D), rigid random moves per
LAMMPS data + dump and GROMACS top + xtc (same coordinates)."""
import numpy as np
from pathlib import Path

def build(smiles='CCCCC', n_side=3, box_A=40.0, n_frames=6, resname='PEN', seed=1, npt=False, npt_amp=0.01):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=7)
    xyz0 = mol.GetConformer().GetPositions()
    xyz0 -= xyz0.mean(0)
    el = [a.GetSymbol() for a in mol.GetAtoms()]
    mass = np.array([a.GetMass() for a in mol.GetAtoms()])
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
    counts = {}
    names = []
    for e in el:
        counts[e] = counts.get(e, 0) + 1
        names.append(f'{e}{counts[e]}')
    n_mol = n_side ** 3
    rng = np.random.default_rng(seed)
    centers = (np.stack(np.meshgrid(*[np.arange(n_side)] * 3, indexing='ij'), -1).reshape(-1, 3) + 0.5) * box_A / n_side
    frames, boxes = [], []
    for f in range(n_frames):
        b = box_A * (1 + (npt_amp * np.sin(f) if npt else 0.0))
        allx = []
        for c in centers:
            q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
            allx.append(xyz0 @ q.T + c * b / box_A + rng.normal(scale=0.5, size=3))
        x = np.concatenate(allx)
        frames.append(np.mod(x, b))                   # wrapped: molecules cut by the box
        boxes.append(b)
    return dict(el=el, mass=mass, bonds=bonds, names=names, n_mol=n_mol, n_at=len(el), frames=frames, boxes=boxes,
                resname=resname)

def _thermo(n, thermo):
    """T (K) and P (atm) per frame: 'ramp' climbs by 1 per frame (an inconsistent file), 'steady' is constant at 300 K
    and 1 atm, 'noisy' fluctuates around them like a thermostatted, barostatted liquid (3 K, 150 atm)."""
    f = np.arange(n)
    if thermo == 'ramp':
        return 300.0 + f, 1.0 + f
    if thermo == 'steady':
        return np.full(n, 300.0), np.full(n, 1.0)
    rng = np.random.default_rng(11)
    return 300.0 + rng.normal(0, 3.0, n), 1.0 + rng.normal(0, 150.0, n)

def write_lammps(s, d, thermo='ramp'):
    d = Path(d); d.mkdir(parents=True, exist_ok=True)
    types = sorted(set(s['el']), key=s['el'].index)
    tmass = {e: s['mass'][s['el'].index(e)] for e in types}
    n, nm = s['n_at'], s['n_mol']
    L = ['test', '', f'{n * nm} atoms', f'{len(s["bonds"]) * nm} bonds', f'{len(types)} atom types', '1 bond types', '',
         f'0 {s["boxes"][0]} xlo xhi', f'0 {s["boxes"][0]} ylo yhi', f'0 {s["boxes"][0]} zlo zhi', '', 'Masses', '']
    L += [f'{i + 1} {tmass[e]}' for i, e in enumerate(types)] + ['', 'Atoms # full', '']
    x0 = s['frames'][0]
    for m in range(nm):
        for i in range(n):
            k = m * n + i
            L.append(f'{k + 1} {m + 1} {types.index(s["el"][i]) + 1} 0.0 {x0[k, 0]:.4f} {x0[k, 1]:.4f} {x0[k, 2]:.4f}')
    L += ['', 'Bonds', '']
    c = 0
    for m in range(nm):
        for a, b in s['bonds']:
            c += 1
            L.append(f'{c} 1 {m * n + a + 1} {m * n + b + 1}')
    (d/'system.data').write_text('\n'.join(L) + '\n')
    D = []
    for f, (x, b) in enumerate(zip(s['frames'], s['boxes'])):
        D += ['ITEM: TIMESTEP', str(f * 1000), 'ITEM: NUMBER OF ATOMS', str(n * nm), 'ITEM: BOX BOUNDS pp pp pp']
        D += [f'0 {b}'] * 3 + ['ITEM: ATOMS id type x y z']
        D += [f'{k + 1} {types.index(s["el"][k % n]) + 1} {x[k, 0]:.4f} {x[k, 1]:.4f} {x[k, 2]:.4f}' for k in range(n * nm)]
    (d/'traj.lammpstrj').write_text('\n'.join(D) + '\n')
    log = ['units real', 'timestep 1.0', 'Step Temp Press Volume']
    T, P = _thermo(len(s['boxes']), thermo)
    log += [f'{f * 1000} {T[f]} {P[f]} {b ** 3}' for f, b in enumerate(s['boxes'])] + ['Loop time of 1.0']
    (d/'log.lammps').write_text('\n'.join(log) + '\n')

def write_gromacs(s, d, thermo='ramp'):
    import MDAnalysis as mda
    d = Path(d); d.mkdir(parents=True, exist_ok=True)
    n, nm = s['n_at'], s['n_mol']
    T = ['[ defaults ]', '1 2 yes 0.5 0.5', '', '[ atomtypes ]']
    T += [f'{e} {s["mass"][s["el"].index(e)]:.4f} 0.0 A 0.3 0.3' for e in sorted(set(s['el']), key=s['el'].index)]
    T += ['', '[ moleculetype ]', 'MOL 3', '', '[ atoms ]']
    T += [f'{i + 1} {s["el"][i]} 1 {s["resname"]} {s["names"][i]} {i + 1} 0.0 {s["mass"][i]:.4f}' for i in range(n)]
    T += ['', '[ bonds ]'] + [f'{a + 1} {b + 1} 1' for a, b in s['bonds']]
    T += ['', '[ system ]', 'test', '', '[ molecules ]', f'MOL {nm}']
    (d/'system.top').write_text('\n'.join(T) + '\n')
    u = mda.Universe.empty(n * nm, trajectory=True)
    with mda.Writer(str(d/'traj.xtc'), n * nm) as w:
        for f, (x, b) in enumerate(zip(s['frames'], s['boxes'])):
            u.atoms.positions = x
            u.dimensions = [b, b, b, 90, 90, 90]
            u.trajectory.ts.time = f * 1.0
            w.write(u.atoms)
    xv = ['@ s0 legend "Temperature"', '@ s1 legend "Pressure"', '@ s2 legend "Volume"']
    T, P = _thermo(len(s['boxes']), thermo)
    xv += [f'{f * 1.0} {T[f]} {P[f] * 1.01325} {(b / 10) ** 3}' for f, b in enumerate(s['boxes'])]
    (d/'energy.xvg').write_text('\n'.join(xv) + '\n')
