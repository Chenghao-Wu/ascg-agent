"""AutoPoly v2.0 OPLS-AA backend (real upstream three-stage API).

AutoPoly is an external dependency, NOT reimplemented or bundled. This adapter
constructs AA topology and force-field artifacts, but does not pretend that a
LAMMPS simulation has run. For exact paper reproduction, user-verified
complement SMILES, sequence, stereochemistry, and mapping are prerequisites.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import hashlib
import json
import re


class AutoPolyIntegrationError(RuntimeError):
    """Missing dependency, unsupported input, or failed physical artifact gate."""


@dataclass(frozen=True)
class PolymerRecipe:
    """One AutoPoly Polymer definition. Each chain may have its own recipe.

    End-capped complement SMILES are specified explicitly; no guessing
    polymer chemistry from its name. For a random copolymer, create separate
    recipes for the actual sampled sequences (chain_num=1 per unique sequence).
    """
    sequence: tuple[str, ...]
    chain_num: int = 20
    topology: str = "linear"
    tacticity: str = "atactic"

    def __post_init__(self):
        if self.chain_num <= 0:
            raise ValueError("chain_num must be positive")
        if not self.sequence:
            raise ValueError("Explicit complement SMILES sequence required")
        if self.topology != 'linear':
            raise ValueError('Paper benchmark supports linear chains only')
        if self.tacticity not in {'atactic','isotactic','syndiotactic'}:
            raise ValueError('Invalid tacticity')
        if len(self.sequence) > 1:
            counts = [x.count('[*]') for x in self.sequence]
            if counts != [1]+[2]*(len(self.sequence)-2)+[1]:
                raise ValueError('AutoPoly v2 linear sequence requires 1/2/.../2/1 wildcards')
        elif '[*]' in self.sequence[0]:
            raise ValueError('Single monomer must be capped without wildcard')


@dataclass(frozen=True)
class OPLSSystemRecipe:
    name: str
    output_root: str
    polymers: tuple[PolymerRecipe, ...]
    seed: int = 42
    placement: str = 'mc_random'
    # A 0.9 g/cm3 initialization is a MASS density, not AutoPoly's
    # monomer_density (monomers/A^3). A checked explicit box length must be
    # computed with the true chemical composition/masses after generation.
    box_lengths_angstrom: tuple[float, float, float] | None = None
    initial_density_g_cm3: float | None = 0.9

    def __post_init__(self):
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', self.name):
            raise ValueError('Name must be a safe filesystem label')
        if not self.polymers:
            raise ValueError('At least one polymer recipe required')
        if self.initial_density_g_cm3 is not None and self.initial_density_g_cm3 <= 0:
            raise ValueError('Initial mass density must be positive')
        if self.box_lengths_angstrom is not None and self.initial_density_g_cm3 is not None:
            raise ValueError('Specify either a mass density or explicit box, not both')
        if self.box_lengths_angstrom is not None:
            if len(self.box_lengths_angstrom) != 3 or min(self.box_lengths_angstrom) <= 0:
                raise ValueError('Positive three-dimensional explicit box required')


def paper_pe_recipe(output_root: str, seed: int=42) -> OPLSSystemRecipe:
    """Validated syntax from AutoPoly v2 example, 20 x 10 PE reference task.

    This does NOT certify identical generated coordinates, AA charge types,
    or equilibrium density to the paper; its unpublished seed is unknown.
    """
    return OPLSSystemRecipe(
        name='pe', output_root=output_root,
        polymers=(PolymerRecipe(
            sequence=('CC[*]',)+('[*]CC[*]',)*8+('[*]CC',),
            chain_num=20, tacticity='atactic'),), seed=seed)


def beads_per_chain(task_key: str, recipes: dict, mappings: dict) -> float:
    """CG beads per chain from the unit sequence and the repeat-unit groups of the mapping (mean for random
    copolymers)."""
    task = recipes['tasks'][task_key]
    spec = mappings['tasks'][task_key]['units']

    def n_beads(code, pos):
        b = spec[code]
        return len(b[pos] if isinstance(b, dict) else b)
    if 'random' in task:
        tot = sum(task['random'].values())
        mean = {pos: sum(w / tot * n_beads(c, pos) for c, w in task['random'].items())
                for pos in ('first', 'middle', 'last')}
        return mean['first'] + (task['length'] - 2) * mean['middle'] + mean['last']
    tokens = []
    for token in task['sequence'].split():
        code, _, count = token.partition('*')
        tokens += [code] * int(count or 1)
    return float(sum(n_beads(c, 'first' if i == 0 else 'last' if i == len(tokens) - 1 else 'middle')
                     for i, c in enumerate(tokens)))


def chains_for_beads(task_key: str, recipes: dict, mappings: dict, *, target_beads: int = 1000,
                     min_chains: int = 20) -> int:
    """Chains giving ~target_beads CG beads (default 1000 beads, >= 20 chains)."""
    return max(min_chains, int(round(target_beads / beads_per_chain(task_key, recipes, mappings))))


def study_recipe(task_key: str, output_root: str, *, recipes: dict, chains: int = 80,
                 seed: int = 42) -> OPLSSystemRecipe:
    """OPLSSystemRecipe for one task of a recipes file (the benchmark's: benchmarks/aa_recipes.json).

    Sequences are unit codes ('E*10' or 'S S BD ...'); position 0 takes the unit's
    'first' SMILES, the last position its 'last', the rest 'middle'. A 'random'
    task draws an independent sequence per chain from `seed`, one chain per
    PolymerRecipe. The box is dilute and RadonPy21 compresses it, so no
    initial mass density is imposed: AutoPoly's mc_random packer places each
    chain as a sphere of radius 4*sqrt(dop/6)*2 + 2 A (geometry.py, offset 4)
    and falls back to a grid of spacing 2r, so the cube edge is (ceil(N^(1/3)) + 2)
    sphere diameters plus a 10 A margin (its own box estimate fails for 80 chains).
    """
    import math
    import random
    task = recipes['tasks'][task_key]
    units = recipes['units']
    tacticity = task.get('tacticity', recipes['defaults']['tacticity'])
    if 'molecules' in task:                # molecular liquid / mixture: one capped molecule per AutoPoly 'polymer'
        polymers = tuple(PolymerRecipe((m['smiles'],), chain_num=int(m['count'])) for m in task['molecules'])
        n = sum(p.chain_num for p in polymers)
        radius = 4.0 * math.sqrt(1 / 6) * 2 + 2
        side = (math.ceil(round(n ** (1 / 3), 9)) + 2) * 2 * radius + 10.0
        return OPLSSystemRecipe(task_key, output_root, polymers, seed=seed, initial_density_g_cm3=None,
                                box_lengths_angstrom=(side, side, side))
    chains = int(task.get('chains', chains))

    def smiles(codes):
        if len(codes) < 2:
            raise ValueError('Linear chains need at least two units')
        if any(c not in units for c in codes):
            raise ValueError(f'Unknown unit code in {codes}')
        return tuple(units[c]['first' if i == 0 else 'last' if i == len(codes)-1 else 'middle']
                     for i, c in enumerate(codes))

    if 'random' in task:
        rng = random.Random(seed)
        codes, weights = zip(*task['random'].items())
        polymers = tuple(PolymerRecipe(smiles(rng.choices(codes, weights, k=task['length'])),
                                       chain_num=1, tacticity=tacticity)
                         for _ in range(chains))
    else:
        tokens = []
        for token in task['sequence'].split():
            code, _, count = token.partition('*')
            tokens += [code] * int(count or 1)
        polymers = (PolymerRecipe(smiles(tokens), chain_num=chains, tacticity=tacticity),)
    dop = max(len(p.sequence) for p in polymers)
    radius = 4.0 * math.sqrt(dop / 6) * 2 + 2
    # AutoPoly's fallback grid holds only floor-many 2r cells inside the margins (3^3 for a
    # 5-cell cube: 100 chains failed), so two spare cells per edge: >= ceil(N^1/3)^3 positions.
    # The resulting dilute box (~0.01 g/cm3) is the large start box the GROMACS EM relaxes in.
    side = (math.ceil(round(chains ** (1 / 3), 9)) + 2) * 2 * radius + 10.0
    return OPLSSystemRecipe(task_key, output_root, polymers, seed=seed, initial_density_g_cm3=None,
                            box_lengths_angstrom=(side, side, side))


def box_from_geometry(geometry_json: str | Path, density_g_cm3: float) -> tuple[float,float,float]:
    """Calculate a cubic AA box from all full-chain masses (not bead counts).

    AutoPoly 'monomer_density' is a number density, not the 0.9 g/cm3
    mass density in the reference paper. The stored, fully connected
    chain graphs include real end groups, avoiding an idealized repeat-mass
    error for the short 10-monomer chains in the benchmark.
    """
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    data=json.loads(Path(geometry_json).read_text())
    graphs=data['chain_graphs']
    total_g_mol=0.0
    if not data.get('chains') or data.get('molecules'):
        raise AutoPolyIntegrationError('Expect pure polymer chains for paper mass-density box sizing')
    for ch in data['chains']:
        mid=ch['model_id']
        m=Chem.MolFromSmiles(graphs[mid]['smiles_mapped'])
        if m is None:raise AutoPolyIntegrationError(f'Cannot parse AutoPoly chain graph {mid}')
        total_g_mol += Descriptors.MolWt(m)
    if density_g_cm3<=0 or total_g_mol<=0:raise ValueError('Invalid density or polymer mass')
    side_A=(total_g_mol/(6.02214076e23*density_g_cm3)*1e24)**(1/3)
    return (side_A,side_A,side_A)


def _hash(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate_opls_aa(recipe: OPLSSystemRecipe, *, pack: bool=True) -> dict:
    """Execute upstream GeometryBuilder -> UnitTyper -> BoxPacker.

    A caller must provide AutoPoly v2.0 on PYTHONPATH. `pack=False` is a
    real topology/OPLS assignment run that skips moltemplate; it is useful
    for performing a force-field audit before any AA MD.
    """
    try:
        import AutoPoly
        from AutoPoly import (System, Polymer, GeometryBuilder, GeometryConfig,
                              UnitTyper, BoxPacker)
    except (ImportError, AttributeError) as exc:
        raise AutoPolyIntegrationError(
            'Install AutoPoly v2.0 (pip install -e /path/to/AutoPoly); '
            'it is not bundled in the framework') from exc
    version = getattr(AutoPoly,'__version__','unknown')
    if not str(version).startswith('2.0'):
        raise AutoPolyIntegrationError(f'Expected AutoPoly v2.0, got {version!r}')
    out = Path(recipe.output_root).expanduser().resolve()
    system = System(out=str(out))
    # AutoPoly draws tacticity in Polymer.__init__ from Python's unseeded `random`
    # (GeometryConfig.rng_seed only seeds numpy), so seed it per recipe.
    import random
    ps=[]
    for i, p in enumerate(recipe.polymers):
        random.seed(recipe.seed * 100003 + i)
        ps.append(Polymer(chain_num=p.chain_num, sequence=list(p.sequence),
                          topology=p.topology, tacticity=p.tacticity))
    geometry=GeometryBuilder(system, recipe.name,
              GeometryConfig(use_mc_chain_growth=True, rng_seed=recipe.seed)).build(ps)
    geo=Path(geometry.dir)/'geometry.json'
    if not geo.is_file():
        raise AutoPolyIntegrationError(f'AutoPoly missing geometry.json: {geo}')
    library=UnitTyper(geometry.dir,'oplsaa').type()
    build_dir=out/recipe.name/'build'/'oplsaa'
    unit_manifest=build_dir/'units.json'
    if not unit_manifest.is_file() or library.force_field!='oplsaa':
        raise AutoPolyIntegrationError('AutoPoly OPLS unit typing did not produce valid units.json')
    library.validate(build_dir)
    result={
        'source_backend':'WuGroup-XJTLU/AutoPoly',
        'source_branch':'v2.0',
        'version':str(version),
        'force_field':'oplsaa',
        'name':recipe.name,'seed':recipe.seed,
        'total_chains':sum(p.chain_num for p in recipe.polymers),
        'polymer_lengths':sorted(set(len(p.sequence) for p in recipe.polymers)),
        'geometry_json':str(geo),'units_json':str(unit_manifest),
        'inputs':{'polymer_sequences':[
            {'chain_num':p.chain_num,'sequence':list(p.sequence),'tacticity':p.tacticity,
             'python_random_seed':recipe.seed * 100003 + i}
            for i, p in enumerate(recipe.polymers)]},
        'file_sha256':{str(geo):_hash(geo),str(unit_manifest):_hash(unit_manifest)},
        'packing_status':'not_requested',
    }
    if pack:
        kwargs=dict(strategy=recipe.placement,rng_seed=recipe.seed)
        if recipe.box_lengths_angstrom is not None:
            kwargs['box_dims']=recipe.box_lengths_angstrom
        elif recipe.initial_density_g_cm3 is not None:
            kwargs['box_dims']=box_from_geometry(geo,recipe.initial_density_g_cm3)
        BoxPacker(system,recipe.name,**kwargs).pack(library)
        # AutoPoly v2 mv_files() copies production files into project root.
        project=out/recipe.name
        expected=['system.data','system.in.init','system.in.settings']
        missing=[f for f in expected if not (project/f).is_file()]
        if missing:
            raise AutoPolyIntegrationError(f'AutoPoly packing missing {missing} in {project}')
        from .lammps_audit import audit_lammps_data
        audit=audit_lammps_data(project/'system.data', expected_density=recipe.initial_density_g_cm3)
        result.update({
            'lammps_audit':audit.to_dict(),
            'initial_density_target_g_cm3':recipe.initial_density_g_cm3,
            'requested_box_lengths_angstrom':list(kwargs.get('box_dims') or []),
            'packing_status':'complete',
            'lammps_data':str(project/'system.data'),
            'lammps_init':str(project/'system.in.init'),
            'lammps_settings':str(project/'system.in.settings'),
            'lammps_charges':str(project/'system.in.charges') if (project/'system.in.charges').is_file() else None,
        })
        for f in expected:
            result['file_sha256'][str(project/f)]=_hash(project/f)
    output=out/recipe.name/'cgmas_autopoly_provenance.json'
    output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    result['provenance']=str(output)
    return result
