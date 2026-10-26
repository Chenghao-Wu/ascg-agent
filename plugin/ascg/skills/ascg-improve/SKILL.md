---
name: ascg-improve
description: Maintainer skill: learn from agent and operator transcripts - mine them with ascg trace mine, find repeated ad-hoc work, new failure signatures and errors, and draft tool, skill and test updates for review.
---
# Learn from trajectories

1. `ascg trace mine --transcript <session.jsonl> --out summary.json` (Claude Code transcripts live under
   `~/.claude/projects/<project>/`): tool use, ascg calls, other shell commands, errors, skills, subagents, user turns.
   Also read the study's `calls.jsonl` and `events.jsonl`.
2. Look for: repeated ad-hoc analyses or shell commands (-> a new ascg tool), new failure signatures and how they were
   resolved (-> a case in ascg-diagnose / aa-simulate references), tool errors (-> a policy or a better message),
   user corrections (-> a rule in a skill), long waits (-> events).
3. Draft the change on a branch: tool + test, skill text, a replayable fixture (the task directory of the case). Never
   rewrite skills or policies automatically; a person reviews every change.
