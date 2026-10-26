"""The guides, READMEs and skills name only tools and parameters that exist (stale docs fail here)."""
import re
from pathlib import Path

from ascg import tools  # noqa: F401  (registers the tools)
from ascg.registry import TOOLS

ROOT = Path(__file__).resolve().parents[2]
DOCS = [*sorted((ROOT/'docs'/'guide').glob('*.md')), ROOT/'eval'/'README.md', ROOT/'README.md',
        *sorted((ROOT/'plugin'/'ascg'/'skills').glob('*/SKILL.md')), *sorted((ROOT/'plugin'/'ascg'/'skills').glob('*/references/*.md')),
        *[p for p in (ROOT/'tools'/'public_export'/'overlay').rglob('*.md')]]
SPECIAL = {('init', 'codex'): {'workspace', 'studies', 'profile', 'codex_home', 'force', 'dry_run'}}
CMD = re.compile(r'\bascg[ \t]+([a-z][a-z-]*)[ \t]+([a-z][a-z-]*)((?:[ \t]+(?!ascg\b)(?:--[a-z][a-z0-9-]*(?:=\S+)?|[^\s`|;#]+))*)')


def _commands(text):
    text = re.sub(r'\\\n\s*', ' ', text)                               # shell line continuations
    for m in CMD.finditer(text):
        yield m.group(1), m.group(2), re.findall(r'--([a-z][a-z0-9-]*)', m.group(3) or '')


def test_documented_commands_exist():
    from ascg_core.cli import COMMANDS
    groups = {g for g, _ in TOOLS} | {'init'}
    bad = []
    for doc in DOCS:
        for g, a, opts in _commands(doc.read_text()):
            if g not in groups or g in COMMANDS:
                continue                                                  # prose ("ascg runs ..."), project CLI
            if (g, a) in SPECIAL:
                params = SPECIAL[(g, a)]
            elif (g, a) in TOOLS:
                t = TOOLS[(g, a)]
                params = set(t.params)
            else:
                bad.append(f'{doc.relative_to(ROOT)}: ascg {g} {a}')
                continue
            for o in opts:
                k = o.replace('-', '_')
                if k in params or k in ('help', 'json') or (k.startswith('no_') and k[3:] in params):
                    continue
                bad.append(f'{doc.relative_to(ROOT)}: ascg {g} {a} --{o}')
    assert not bad, '\n'.join(sorted(set(bad)))


def test_guides_link_to_files_that_exist():
    bad = []
    for doc in [*sorted((ROOT/'docs'/'guide').glob('*.md')), ROOT/'eval'/'README.md']:
        text = re.sub(r'`[^`]*`', '', re.sub(r'```.*?```', '', doc.read_text(), flags=re.S))   # code is not a link
        for link in re.findall(r'\]\(([^)#\s]+)(?:#[^)]*)?\)', text):
            if not link.startswith(('http://', 'https://')) and not (doc.parent/link).exists():
                bad.append(f'{doc.relative_to(ROOT)} -> {link}')
    assert not bad, '\n'.join(bad)
