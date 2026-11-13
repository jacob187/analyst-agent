"""Benchmark harness: run a fixed question set through an agent architecture,
measure cost/speed from OpenTelemetry spans, compare two runs with a blind LLM
judge, and write JSON (for agents/gates) plus a PDF (for the human owner).

CLI: ``uv run python -m evals.benchmark {run,judge,summarize,report}``.
"""
