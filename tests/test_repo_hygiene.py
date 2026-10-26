"""The repository holds no machine-specific paths or personal contact details (generic patterns only; a site
checklist is kept outside the repository)."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = [r'/Users/[A-Za-z]', r'/gpfs/', r'/home/(?!USER\b|user\b)[a-z][a-z0-9_]+/', r'[\w.+-]+@(gmail|qq|163|outlook)\.com']
SKIP = {'.git', '__pycache__', '.pytest_cache', 'build', 'dist', 'runs'}


def test_no_machine_paths_or_personal_contacts():
    pats = [re.compile(p) for p in PATTERNS]
    hits = []
    for f in ROOT.rglob('*'):
        if not f.is_file() or SKIP & set(f.relative_to(ROOT).parts) or f.suffix in {'.png', '.npz', '.pdf'}:
            continue
        try:
            text = f.read_text()
        except (UnicodeDecodeError, OSError):
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if f.name == 'test_repo_hygiene.py':
                continue
            if any(p.search(line) for p in pats):
                hits.append(f'{f.relative_to(ROOT)}:{i}: {line.strip()[:120]}')
    assert not hits, 'machine-specific paths or personal contacts:\n' + '\n'.join(hits)
