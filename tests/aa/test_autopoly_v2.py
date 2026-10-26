import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
import pytest

from ascg_core.aa.autopoly_v2 import (PolymerRecipe, OPLSSystemRecipe, box_from_geometry,
                                       generate_opls_aa, paper_pe_recipe, AutoPolyIntegrationError)


def test_paper_pe_recipe():
    spec=paper_pe_recipe('/tmp/test_auto')
    assert len(spec.polymers[0].sequence)==10
    assert spec.polymers[0].chain_num==20
    assert spec.initial_density_g_cm3==0.9


def test_complement_smiles_contract():
    PolymerRecipe(('CC[*]','[*]CC[*]','[*]CC'),chain_num=20)
    with pytest.raises(ValueError,match='wildcards'):
        PolymerRecipe(('CC[*]','[*]CC','[*]CC'))
    with pytest.raises(ValueError,match='linear'):
        PolymerRecipe(('[*]CC[*]',),topology='ring')


def test_initial_density_is_mass_not_number_density(tmp_path):
    pytest.importorskip('rdkit')
    # 20 explicit ethane molecules C2H6 (30.07 g/mol) at 0.9 g/cm3
    # The total mass, not number of repeat units, sets V.
    path=tmp_path/'geometry.json'
    path.write_text(json.dumps(dict(chain_graphs={'model_0':{'smiles_mapped':'CC'}},
                                   chains=[{'model_id':'model_0'}]*20,molecules=[])))
    side=box_from_geometry(path,0.9)[0]
    assert side==pytest.approx((20*30.07/(6.02214076e23*0.9)*1e24)**(1/3),rel=1e-2)


def test_recipe_does_not_silently_choose_inconsistent_box():
    p=PolymerRecipe(('CC[*]','[*]CC'),chain_num=1)
    with pytest.raises(ValueError,match='either'):
        OPLSSystemRecipe('x','/tmp',(p,),box_lengths_angstrom=(30.,30.,30.))


def test_real_autopoly_api_stages_with_stub(monkeypatch,tmp_path):
    """Contract check, NOT a physical run or a replacement for AutoPoly tests."""
    mod=ModuleType('AutoPoly')
    mod.__version__='2.0.0'
    actions=[]
    class System:
        def __init__(self,out):self.out=Path(out)
    class Polymer:
        def __init__(self,**kw):actions.append(('Polymer',kw))
    class GeometryConfig:
        def __init__(self,**kw):self.kw=kw
    class GeometryBuilder:
        def __init__(self,system,name,config):self.path=system.out/name/'geometry'
        def build(self,polymers):
            self.path.mkdir(parents=True,exist_ok=True)
            (self.path/'geometry.json').write_text(json.dumps(dict(
                chains=[{'model_id':'model_0'}]*20,
                chain_graphs={'model_0':{'smiles_mapped':'CCCC'}},molecules=[])))
            actions.append(('geometry',len(polymers)))
            return SimpleNamespace(dir=str(self.path))
    class Library:
        force_field='oplsaa'
        def validate(self,dir):actions.append(('validation',str(dir)))
    class UnitTyper:
        def __init__(self,geometry_dir,ff):
            self.dir=Path(geometry_dir).parent/'build'/ff
        def type(self):
            self.dir.mkdir(parents=True,exist_ok=True)
            (self.dir/'units.json').write_text('{}')
            actions.append(('typing','oplsaa'))
            return Library()
    class BoxPacker:
        def __init__(self,system,name,**kwargs):
            self.project=system.out/name
            actions.append(('packer_options',kwargs))
        def pack(self,lib):
            for name in ('system.data','system.in.init','system.in.settings'):
                (self.project/name).write_text(name)
            (self.project/'system.data').write_text('''LAMMPS data

2 atoms
1 bonds
1 atom types

0 10 xlo xhi
0 10 ylo yhi
0 10 zlo zhi

Masses

1 12

Atoms # full

1 1 1 0.0 1.0 2.0 3.0
2 1 1 0.0 2.0 2.0 3.0

Bonds

1 1 1 2
''')
            actions.append(('packing','done'))
    for x in (System,Polymer,GeometryConfig,GeometryBuilder,UnitTyper,BoxPacker):
        setattr(mod,x.__name__,x)
    monkeypatch.setitem(__import__('sys').modules,'AutoPoly',mod)
    recipe=OPLSSystemRecipe(name='pe',output_root=str(tmp_path),polymers=paper_pe_recipe(str(tmp_path)).polymers,initial_density_g_cm3=None,box_lengths_angstrom=(10.,10.,10.))
    result=generate_opls_aa(recipe)
    assert result['packing_status']=='complete'
    assert result['requested_box_lengths_angstrom'][0]>0
    assert result['total_chains']==20
    assert [a[0] for a in actions].count('typing')==1
    assert Path(result['provenance']).is_file()


def test_study_recipes_cover_the_benchmark():
    from ascg_core.aa.autopoly_v2 import study_recipe
    root=Path(__file__).resolve().parents[2]/'benchmarks'
    recipes=json.loads((root/'aa_recipes.json').read_text())
    systems=json.loads((root/'ascg_library.json').read_text())['systems']
    assert sorted(s['recipe']['task'] for s in systems)==sorted(recipes['tasks'])
    for k,t in recipes['tasks'].items():
        if t['class']!='polymer':
            continue
        r=study_recipe(k,'/tmp/x',recipes=recipes)
        assert sum(p.chain_num for p in r.polymers)==t['chains'] and r.initial_density_g_cm3 is None
    sbs=study_recipe('sbs','/tmp/x',recipes=recipes).polymers[0].sequence
    assert sbs[3]=='[*]CC=CC[*]' and sbs[-1]=='[*]CC(c1ccccc1)'
    ep1=study_recipe('ep','/tmp/x',recipes=recipes,seed=7)
    ep2=study_recipe('ep','/tmp/x',recipes=recipes,seed=7)
    assert len(ep1.polymers)==recipes['tasks']['ep']['chains'] and ep1==ep2   # one random sequence per chain
    assert len({p.sequence for p in ep1.polymers})>1
