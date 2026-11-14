"""``report``: a PDF for the human owner (JSON stays the machine interface).

Layout: summary tables first, then one page-group per question with each run's
answer side by side plus the judge verdict. Pure Python (fpdf2, core fonts), so
text is reduced to latin-1 and markdown is flattened to readable plain text.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from fpdf import FPDF

from evals.benchmark.pricing import estimate_cost
from evals.benchmark.runs import LoadedRun
from evals.benchmark.schemas import Judgement, JudgedQuestion, QuestionResult

_FONT = "Helvetica"
_LINE_H = 4.4
_GAP = 5.0

_CHAR_MAP = {
    "‘": "'", "’": "'", "‚": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "−": "-", "•": "-", "‣": "-",
    "·": "-", "…": "...", "→": "->", "←": "<-", "≥": ">=",
    "≤": "<=", "≈": "~", "≠": "!=", "×": "x", " ": " ",
    " ": " ", " ": " ", " ": " ", " ": " ", "​": "",
    "✓": "ok", "✔": "ok", "✗": "x", "▲": "^", "▼": "v",
    "\t": "    ",
}
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
_HEADER_RE = re.compile(r"^\s{0,3}#{1,6}\s*")
_BULLET_RE = re.compile(r"^(\s*)[*+]\s+")
_TABLE_RULE_RE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")


def pdf_text(text: str) -> str:
    """Reduce text to what a core PDF font can draw (latin-1; others become '?')."""
    for src, dst in _CHAR_MAP.items():
        text = text.replace(src, dst)
    text = _CONTROL_RE.sub("", text)
    return text.encode("latin-1", errors="replace").decode("latin-1")


def flatten_markdown(text: str) -> str:
    """Markdown to readable plain text: no header/emphasis markers, bullets kept."""
    out: list[str] = []
    for line in text.replace("\r\n", "\n").split("\n"):
        if _TABLE_RULE_RE.match(line) and "-" in line:
            continue
        line = _HEADER_RE.sub("", line)
        line = _BULLET_RE.sub(r"\1- ", line)
        line = _LINK_RE.sub(r"\1 (\2)", line)
        line = line.replace("**", "").replace("__", "").replace("`", "")
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            line = " | ".join(c.strip() for c in stripped.strip("|").split("|"))
        out.append(line.rstrip())
    return "\n".join(out)


def _clean(text: str) -> str:
    return pdf_text(flatten_markdown(text))


def _cost_text(value: float | None) -> str:
    return "unknown" if value is None else f"${value:.4f}"


def _num(value: float | None, fmt: str) -> str:
    return "-" if value is None else format(value, fmt)


class _Pdf(FPDF):
    def footer(self) -> None:
        self.set_y(-10)
        self.set_font(_FONT, "I", 8)
        self.cell(0, 5, f"Page {self.page_no()}", align="C")


def _heading(pdf: FPDF, text: str, size: int = 13) -> None:
    pdf.set_font(_FONT, "B", size)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(0, size * 0.5, pdf_text(text), new_x="LMARGIN", new_y="NEXT")


def _paragraph(pdf: FPDF, text: str, *, size: int = 9, style: str = "") -> None:
    pdf.set_font(_FONT, style, size)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(0, _LINE_H, _clean(text), align="L", new_x="LMARGIN", new_y="NEXT")


def _wrap(pdf: FPDF, text: str, width: float) -> list[str]:
    lines = pdf.multi_cell(
        width, _LINE_H, text, align="L", dry_run=True, output="LINES", wrapmode="WORD"
    )
    return list(lines)


def _columns(pdf: FPDF, headers: list[str], texts: list[str]) -> None:
    """Side-by-side wrapped text, paginating line by line (answers can exceed a page)."""
    n = len(texts)
    width = (pdf.epw - _GAP * (n - 1)) / n

    def draw_header() -> None:
        pdf.set_font(_FONT, "B", 9)
        for c, header in enumerate(headers):
            pdf.set_x(pdf.l_margin + c * (width + _GAP))
            pdf.cell(width, _LINE_H + 1, _fit(pdf, pdf_text(header), width))
        pdf.ln(_LINE_H + 1)
        pdf.set_font(_FONT, "", 9)

    pdf.set_font(_FONT, "", 9)
    wrapped = [_wrap(pdf, _clean(t), width) for t in texts]
    if pdf.will_page_break(_LINE_H * 4):
        pdf.add_page()
    draw_header()
    for i in range(max((len(w) for w in wrapped), default=0)):
        if pdf.will_page_break(_LINE_H):
            pdf.add_page()
            draw_header()
        for c, lines in enumerate(wrapped):
            pdf.set_x(pdf.l_margin + c * (width + _GAP))
            pdf.cell(width, _LINE_H, lines[i] if i < len(lines) else "")
        pdf.ln(_LINE_H)


def _fit(pdf: FPDF, text: str, width: float) -> str:
    """Truncate to one line of ``width`` (headers must not wrap)."""
    if pdf.get_string_width(text) <= width - 2:
        return text
    while len(text) > 1 and pdf.get_string_width(text + "...") > width - 2:
        text = text[:-1]
    return text + "..."


def _table(pdf: FPDF, header: list[str], rows: list[list[str]], widths: list[float]) -> None:
    pdf.set_font(_FONT, "", 8)
    scale = pdf.epw * 0.995 / sum(widths)  # widths are relative; fill the page width
    widths = [w * scale for w in widths]
    with pdf.table(col_widths=widths, text_align="LEFT", line_height=4.5, width=sum(widths)) as table:
        head = table.row()
        for h in header:
            head.cell(pdf_text(h))
        for values in rows:
            row = table.row()
            for v in values:
                row.cell(pdf_text(v))


def _verdict_lines(
    question_id: str, judgement: Judgement | None, runs: list[LoadedRun]
) -> list[str]:
    if judgement is None:
        return []
    entry: JudgedQuestion | None = next((q for q in judgement.questions if q.id == question_id), None)
    if entry is None:
        return ["Judge verdict: not judged (question not in the judgement)."]

    label_of = {
        "a": "A " + _run_name(judgement.a.run_id, runs, f"({judgement.a.architecture} / {judgement.a.model_id})"),
        "b": "B " + _run_name(judgement.b.run_id, runs, f"({judgement.b.architecture} / {judgement.b.model_id})"),
    }
    if entry.status != "judged":
        return [f"Judge verdict: {entry.status} - {entry.reason or 'no reason recorded'}"]
    winner = "tie" if entry.winner == "tie" else f"{label_of[entry.winner]} wins"
    order = "A shown first" if entry.a_is_first else "B shown first"
    return [
        f"Judge verdict: {winner}  (blind order: {order})",
        f"Unique factual error - {label_of['a']}: {'YES' if entry.a_unique_factual_error else 'no'}; "
        f"{label_of['b']}: {'YES' if entry.b_unique_factual_error else 'no'}",
        f"Rationale: {entry.rationale or ''}",
    ]


def _run_name(run_id: str, runs: list[LoadedRun], fallback: str) -> str:
    for run in runs:
        if run.meta.run_id == run_id:
            return f"[{run.label}]"
    return fallback


def _metrics_line(run: LoadedRun, r: QuestionResult | None) -> str:
    if r is None:
        return f"{run.label}: no result recorded"
    cost = estimate_cost(run.meta.pricing, r.input_tokens, r.output_tokens, r.llm_calls_without_usage)
    tools = " > ".join(t.name for t in r.tool_calls) or "none"
    if len(tools) > 300:
        tools = tools[:297] + "..."
    line = (
        f"{run.label}: {r.status}, {r.llm_calls} LLM calls, {r.input_tokens} in / {r.output_tokens} out tokens, "
        f"{r.latency_s:.1f}s, cost {_cost_text(cost)}; tools: {tools}"
    )
    if r.errors:
        line += f"; errors: {' | '.join(r.errors)[:300]}"
    return line


def render_report(
    runs: list[LoadedRun], judgement: Judgement | None, summary: dict[str, Any]
) -> bytes:
    pdf = _Pdf(orientation="L", unit="mm", format="A4")
    pdf.set_margins(12, 12, 12)
    pdf.set_auto_page_break(auto=True, margin=14)
    pdf.set_title("Analyst Agent benchmark report")
    pdf.add_page()

    _heading(pdf, "Analyst Agent benchmark report", 18)
    as_of = summary["prices_as_of"]
    _paragraph(pdf, f"Generated {summary['generated_at']}.  Prices as of: "
               + (", ".join(as_of) if as_of else "unknown (no price data; costs shown as unknown)"))
    for warning in summary["warnings"]:
        _paragraph(pdf, f"WARNING: {warning}", style="B")
    pdf.ln(2)

    _heading(pdf, "Summary (means per question; cost is the run total)", 12)
    rows = []
    for run in summary["runs"]:
        m, t, q = run["means"], run["totals"], run["questions"]
        j = run["judgement"]
        rows.append([
            run["label"], run["architecture"], run["model_id"],
            f"{q['answered']}/{q['selected']}",
            _num(m["llm_calls"], ".1f"), _num(m["input_tokens"], ".0f"), _num(m["output_tokens"], ".0f"),
            _num(m["latency_s"], ".1f"), _cost_text(t["cost_usd"]),
            "-" if j is None else f"{j['wins']}/{j['ties']}/{j['losses']}",
            "-" if j is None else f"{j['unique_factual_errors']} / {j['opponent_unique_factual_errors']}",
        ])
    _table(
        pdf,
        ["Run", "Architecture", "Model", "Answered", "LLM calls", "In tok", "Out tok",
         "Latency s", "Total cost", "W/T/L", "Unique errors (own / opp)"],
        rows,
        [38, 24, 40, 18, 18, 18, 18, 20, 22, 18, 41],
    )
    pdf.ln(3)

    _heading(pdf, "By category (means per question)", 12)
    cat_rows = []
    for run in summary["runs"]:
        for cat, agg in run["by_category"].items():
            m = agg["means"]
            cat_rows.append([
                run["label"], cat, str(agg["n"]), _num(m["llm_calls"], ".1f"),
                _num(m["total_tokens"], ".0f"), _num(m["latency_s"], ".1f"),
                _cost_text(m["cost_usd"]),
            ])
    _table(
        pdf, ["Run", "Category", "N", "LLM calls", "Tokens", "Latency s", "Mean cost"],
        cat_rows, [60, 40, 14, 24, 28, 26, 30],
    )

    if summary["judgement"] is not None:
        pdf.ln(3)
        jb = summary["judgement"]
        _heading(pdf, "Blind judgement", 12)
        _paragraph(
            pdf,
            f"Judge model: {jb['judge_model']}; shuffle seed: {jb['seed']}. Run A = {jb['a']['dir']} "
            f"({jb['a']['architecture']} / {jb['a']['model_id']}); run B = {jb['b']['dir']} "
            f"({jb['b']['architecture']} / {jb['b']['model_id']}). "
            f"A: {jb['a_counts']['wins']} wins, B: {jb['b_counts']['wins']} wins, "
            f"{jb['a_counts']['ties']} ties, {jb['a_counts']['skipped_or_errored']} not judged.",
        )
        usage, costs = jb["judge_usage"], summary["costs"]
        tokens = (
            "usage not recorded"
            if usage["input_tokens"] is None
            else f"{usage['input_tokens']} in / {usage['output_tokens']} out tokens"
        )
        _paragraph(pdf, f"Judge cost: {_cost_text(usage['cost_usd'])} ({tokens}).", style="B")
        _paragraph(
            pdf,
            f"Run cost: {_cost_text(costs['runs_usd'])}. Runs + judging total: {_cost_text(costs['total_usd'])}.",
        )

    pdf.ln(3)
    _paragraph(
        pdf,
        "Notes: questions run sequentially in one process, so cached filing analyses can make later "
        "questions on the same ticker cheaper. Cost is unknown when a price is missing or any LLM call "
        "reported no token usage.",
        style="I",
    )

    ordered_ids: list[str] = []
    for run in runs:
        for qid in run.meta.question_ids:
            if qid not in ordered_ids:
                ordered_ids.append(qid)

    for qid in ordered_ids:
        pdf.add_page()
        results = [run.questions.get(qid) for run in runs]
        ref = next((r for r in results if r is not None), None)
        _heading(pdf, qid, 14)
        if ref is not None:
            _paragraph(pdf, f"{ref.ticker} | {ref.category}", style="I")
            _paragraph(pdf, f"Question: {ref.question}", size=10, style="B")
        pdf.ln(1)
        for run, r in zip(runs, results):
            _paragraph(pdf, _metrics_line(run, r), size=8)
        pdf.ln(1)
        for line in _verdict_lines(qid, judgement, runs):
            _paragraph(pdf, line, size=9, style="B" if line.startswith("Judge verdict") else "")
        pdf.ln(2)
        _columns(
            pdf,
            [f"{run.label} [{run.meta.architecture} / {run.meta.model_id}]" for run in runs],
            [
                (r.answer if r is not None and r.answer else "(no answer)")
                for r in results
            ],
        )

    return bytes(pdf.output())


def write_report(path: Path | str, runs: list[LoadedRun], judgement: Judgement | None, summary: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(render_report(runs, judgement, summary))
    return path
