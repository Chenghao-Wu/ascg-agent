"""ascg: agentic systematic coarse-graining - deterministic command-line tools for skills and agents.

Local side (standard library only, except `report` and `study verify`, which read the core's protocol and reports
with numpy): tool registry and CLI (`ascg <group> <action>`, JSON in / out), study registry,
compute profiles and transports, policies. Host side (`python -m ascg.remote <verb>`, run through the transport in the
host's environment): wraps the numerical core (ascg_core.ibi) without changing it.
"""
__version__ = '0.6.0'
