"""PreToolUse guard of the ascg plugin (exit 2 + reason on stderr = blocked). Generic rules, no site names:
remote job / process control and deletions only through `ascg`; no credential files; amendments, compute setup
(`compute add` / `deploy`) and agent setup (`ascg init`) are done by a person, not an agent; locked study files and
the policy / gate code are not edited by the agent."""
import json
import re
import sys

BASH_RULES = [
    (r'\bascg\s+amend\s+approve\b', 'amendments are approved by the user at a terminal, not by an agent'),
    (r'\bascg\s+(compute\s+(add|deploy)|init)\b', 'compute setup and agent setup are user actions: give the user the '
     'exact command to run in their terminal'),
    (r'\b(scancel|qdel|bkill|pkill|killall)\b', 'job and process control goes through `ascg run cancel` (own jobs only)'),
    (r'\bssh\b[^|;&]*?(\bsbatch\b|\bqsub\b|\bkill\b|\brm\s+-[a-z]*r|\bmv\s|\bcp\s|\brsync\b)',
     'remote changes go through ascg tools (submit, cancel, sync); read-only ssh is allowed'),
    (r'(\.ssh/|id_rsa|id_ed25519|\.netrc|\bcredentials\b|API_KEY|_TOKEN\b|\.env\b)', 'credential files are off limits'),
    (r'(^|[\s;&|])(\S*/)?fastmd(\s|$)', 'run fastMD through ascg (jobs, selfcheck), not directly'),
]
EDIT_RULES = [
    (r'(/|^)ascg-studies/[^/]+/(protocol\.json|registry\.json|amendments/)', 'study registry, protocol and amendments are '
     'written by ascg tools only'),
    (r'src/ascg/(policy|registry)\.py$', 'policy code is not edited by the agent'),
    (r'src/ascg_core/ibi/(pipeline|analysis|update|runner)\.py$', 'gate and protocol code is not edited by the agent'),
]


def main():
    kind = sys.argv[1] if len(sys.argv) > 1 else 'bash'
    try:
        data = json.load(sys.stdin)
    except json.JSONDecodeError:
        return 0
    ti = data.get('tool_input') or {}
    if kind == 'bash':
        text, rules = ti.get('command', ''), BASH_RULES
    else:
        text, rules = ti.get('file_path') or ti.get('notebook_path') or '', EDIT_RULES
    for pat, why in rules:
        if re.search(pat, text):
            print(f'ascg guard: blocked - {why}', file=sys.stderr)
            return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
