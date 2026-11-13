"""AC5: `report` renders a valid PDF containing every question id."""

import re
import zlib
from pathlib import Path

import pytest

from evals.benchmark.cli import main
from evals.benchmark.judge import judge_runs
from evals.benchmark.report import flatten_markdown, pdf_text, render_report, write_report
from evals.benchmark.runs import load_run, write_model
from evals.benchmark.summary import build_summary
from tests.benchmark.helpers import FIXTURE_QUESTIONS, UNPRICED, FakeJudge, write_run

pytestmark = pytest.mark.eval_unit

IDS = [q[0] for q in FIXTURE_QUESTIONS]

MARKDOWN_ANSWER = """# Risk summary

**Apple** faces supply-chain risk -- see Item 1A; ratios use snake__case__names.

* Revenue grew ≥ 5% — “solid”
* Margin ≈ 44% → stable • €10

| Metric | Value |
|--------|-------|
| EPS | 6.13 |

Source: [10-K](https://example.com/10-k)
"""

# Far longer than a page, plus a token wider than the column.
LONG_ANSWER = "\n".join(f"Paragraph {i}: " + "lorem ipsum dolor sit amet " * 12 for i in range(120))
UNBROKEN = "x" * 400


def _pdf_text_blob(pdf: bytes) -> str:
    """All page-content text: inflate every stream, keep the text-showing operators."""
    chunks = []
    for raw in re.findall(rb"stream\r?\n(.*?)\r?\nendstream", pdf, re.S):
        try:
            chunks.append(zlib.decompress(raw))
        except zlib.error:
            chunks.append(raw)
    blob = b"\n".join(chunks).decode("latin-1")
    return blob.replace("\\(", "(").replace("\\)", ")").replace("\\\\", "\\")  # PDF string escapes


def _page_count(pdf: bytes) -> int:
    return len(re.findall(rb"/Type\s*/Page[^s]", pdf))


@pytest.fixture
async def fixture_report(tmp_path):
    answers_a = {qid: MARKDOWN_ANSWER for qid in IDS}
    answers_a["q-multi"] = LONG_ANSWER
    answers_b = {qid: f"GOOD short answer for {qid}" for qid in IDS}
    answers_b["q-tech"] = UNBROKEN
    answers_b["q-8k"] = None  # failed question
    a = load_run(write_run(tmp_path, "run-a", architecture="arch-alpha", model_id="model-alpha", answers=answers_a))
    b = load_run(
        write_run(tmp_path, "run-b", architecture="arch-beta", model_id="model-beta", answers=answers_b, pricing=UNPRICED)
    )
    judgement = await judge_runs(a, b, FakeJudge(), "judge-model-x", 0, progress=lambda _m: None)
    summary = build_summary([a, b], judgement)
    return [a, b], judgement, summary


class TestReport:
    async def test_is_a_valid_pdf_with_every_question_id(self, fixture_report):
        runs, judgement, summary = fixture_report
        pdf = render_report(runs, judgement, summary)

        assert pdf.startswith(b"%PDF-")
        assert pdf.rstrip().endswith(b"%%EOF")
        text = _pdf_text_blob(pdf)
        for qid in IDS:
            assert qid in text, qid

    async def test_summary_comes_first_and_shows_prices_and_unknown_cost(self, fixture_report):
        runs, judgement, summary = fixture_report
        text = _pdf_text_blob(render_report(runs, judgement, summary))
        assert text.index("Summary") < text.index("q-risk")
        assert "2026-01-15" in text  # prices_as_of
        assert "unknown" in text  # run-b has no prices
        assert "$0.0200" in text  # run-a total cost
        assert "arch-alpha" in text and "model-beta" in text

    async def test_each_question_shows_both_answers_and_the_verdict(self, fixture_report):
        runs, judgement, summary = fixture_report
        text = _pdf_text_blob(render_report(runs, judgement, summary))
        assert "GOOD short answer for q-risk" in text
        assert "Judge verdict" in text
        assert "Rationale: picked" in text
        assert "skipped - no answer in run B" in text  # the failed question
        assert "(no answer)" in text

    async def test_long_answers_paginate_and_unicode_does_not_crash(self, fixture_report):
        runs, judgement, summary = fixture_report
        pdf = render_report(runs, judgement, summary)
        # 5 question sections + summary, and one answer that alone spans several pages.
        assert _page_count(pdf) >= 8
        text = _pdf_text_blob(pdf)
        assert "Paragraph 119" in text  # nothing dropped at the end of the long answer

    async def test_no_judgement_and_single_run(self, tmp_path):
        run = load_run(write_run(tmp_path, "run-a"))
        pdf = render_report([run], None, build_summary([run]))
        assert pdf.startswith(b"%PDF-")
        assert all(qid in _pdf_text_blob(pdf) for qid in IDS)

    async def test_cli_report_writes_to_the_given_path(self, tmp_path, fixture_report):
        runs, judgement, _ = fixture_report
        write_model(tmp_path / "judged" / "judgement.json", judgement)
        out = tmp_path / "nested" / "report.pdf"
        code = main([
            "report", "--runs", str(runs[0].dir), str(runs[1].dir),
            "--judgement", str(tmp_path / "judged"), "--out", str(out),
        ])
        assert code == 0
        assert out.read_bytes().startswith(b"%PDF-")

    async def test_cli_report_defaults_to_benchmark_reports_folder(self, tmp_path, monkeypatch):
        run_dir = write_run(tmp_path, "run-a")
        monkeypatch.chdir(tmp_path)
        assert main(["report", "--runs", str(run_dir)]) == 0
        (pdf,) = Path("benchmark_reports").glob("benchmark-*.pdf")
        assert pdf.read_bytes().startswith(b"%PDF-")

    async def test_write_report_creates_parent_dirs(self, tmp_path):
        run = load_run(write_run(tmp_path, "run-a"))
        path = write_report(tmp_path / "a" / "b" / "r.pdf", [run], None, build_summary([run]))
        assert path.exists()


class TestTextHelpers:
    def test_pdf_text_maps_common_unicode_and_replaces_the_rest(self):
        assert pdf_text("≥ — “q” → 中") == ">= - \"q\" -> ?"

    def test_flatten_markdown_drops_markup_but_keeps_bullets_and_links(self):
        flat = flatten_markdown(MARKDOWN_ANSWER)
        assert "**" not in flat and "# " not in flat and "`" not in flat
        assert "- Revenue grew" in flat
        assert "|--" not in flat
        assert "Metric | Value" in flat
        assert "10-K (https://example.com/10-k)" in flat
        assert "-- see Item" in flat  # double hyphen is plain text, not a formatting toggle
