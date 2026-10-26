# Coding agents: Claude Code and Codex

The agent follows the ascg skills (`plugin/ascg/skills`) and works only through the ascg tools. The tools do every
calculation and enforce the rules; the agent proposes, runs, waits, diagnoses within bounded recovery actions and
explains. You make the scientific decisions.

## What stays with you

| Decision or step | How |
|---|---|
| The system, its state points (T, P) and the budget | the agent asks and writes a proposal; a system ascg builds is approved with `system approve` |
| The mapping | the agent proposes and checks it; you approve it in your words (`map approve` records them) |
| AA simulations that would be run for you | the agent shows the plan and cost (`aa submit --dry-run`); your words go into `aa submit --approval`; the target's core-hour cap still applies |
| Compute setup | you run `ascg compute add` and `ascg compute deploy` (the agent gives you the commands; it runs `doctor` itself) |
| Protocol amendments (a rule change) | the agent proposes (`amend propose`); you approve at a terminal: `ascg amend approve --study S --id A01` |

The agent cannot change gates, metrics or outcomes. A study's protocol is locked by its hash at `study init`; tools
refuse anything outside the bounded recovery actions (`run act`: extend, rewind with a smaller timestep or damping,
restart; at most 3 actions and 36 extra iterations per task) and `run escalate` hands a task back to you.

## Claude Code

```bash
claude --plugin-dir plugin/ascg          # from the clone; then: /ascg:study <what to coarse-grain>
```

Or install the plugin from this repository's marketplace (inside Claude Code):

```
/plugin marketplace add Chenghao-Wu/ascg-agent
/plugin install ascg@ascg-agent
```

A marketplace install has no source tree next to it: install the package in a Python on your PATH
([install](install.md)) and the plugin's `ascg` launcher uses it.

The plugin brings:
- 14 skills (study, specify-system, setup-compute, setup-engine, bring-reference, aa-build, aa-simulate, mapping,
  targets, fit-ibi, diagnose, improve, validate, report);
- commands `/ascg:study <request>` and `/ascg:status <study>`;
- subagents `ascg-diagnostician` (classifies a stopped fit) and `ascg-reviewer` (checks claims against the records);
- a guard hook that blocks cluster commands bypassing ascg, credential files, edits of study records and gate code,
  and the user steps above.

Claude Code asks before each shell command. To let the `ascg` commands run without asking, add to
`.claude/settings.json` in your working folder:

```json
{"permissions": {"allow": ["Bash(ascg *)"]}}
```

## Codex

```bash
ascg init codex --workspace ~/cg-work    # skills, AGENTS.md and rules in the workspace; a Codex profile
codex -p ascg -C ~/cg-work               # then ask for what you want to coarse-grain
codex exec -p ascg -C ~/cg-work "..."    # or headless
```

`ascg init codex` writes:
- `~/cg-work/.agents/skills/`: the ascg skills;
- `~/cg-work/AGENTS.md`: how the skills map onto the tools;
- `~/cg-work/.codex/rules/ascg.rules`: commands that may never leave the sandbox (ssh, scheduler, kill, `ascg`);
- `~/.codex/ascg.config.toml`: the profile. It sets the workspace-write sandbox without network, trusts the
  workspace, and starts the ascg MCP server with your Python, this code, your studies root (`~/ascg-studies`, or
  `--studies`), your compute profile and your system library.

Your own `~/.codex/config.toml` is not changed; `-p ascg` layers the profile on top of it.

**Containment.** The agent's shell runs in Codex's sandbox: no network, writes only to the workspace, /tmp and
`$TMPDIR`. The ascg tools are served over MCP from outside the sandbox, so they are the only route to your studies
and compute targets, under the tools' own policies. `init` refuses if anything the server runs or trusts (Python,
code, studies, compute profile, library) lies where the agent can write. The user steps above are not served as
tools. Rerun `ascg init codex` after updating ascg.

## Other agents

Any agent that speaks MCP can use the same tools: `python -m ascg.mcp` serves them over stdio (tool names
`<group>_<action>`, the CLI's parameters), and the skills are plain Markdown. Start the server outside the agent's
sandbox and pin `ASCG_STUDIES`, `ASCG_COMPUTE` and `ASCG_LIBRARY` in its environment, as `ascg init codex` does.

## Studies and the record

- `ASCG_STUDIES` (default `./ascg-studies`; Codex: `~/ascg-studies`) holds one directory per study: the locked
  protocol, `registry.json`, the append-only `events.jsonl`, `calls.jsonl` (every tool call, with who made it),
  mappings, amendments and fetched reports. Never edit them by hand; `ascg study note` appends a note.
- `ascg study status`, `ascg study events`, `ascg study explain --study S --task K` and `ascg study verify` show
  what happened.
