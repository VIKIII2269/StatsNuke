"""Tests for the course's quiz engine and question banks.

Run: uv run pytest docs/learn/tests -q
"""

from __future__ import annotations

import importlib.util
import math
import random
import re
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.stats

LEARN = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("quiz", LEARN / "quiz.py")
assert spec is not None
assert spec.loader is not None
quiz = importlib.util.module_from_spec(spec)
sys.modules["quiz"] = quiz
spec.loader.exec_module(quiz)

BANKS = quiz.load_banks()
ALL = [q for b in BANKS.values() for q in b.questions]


def q(**kw: object) -> quiz.Question:
    base: dict[str, object] = {
        "id": "qx-01",
        "module": "99",
        "type": "mcq",
        "prompt": "?",
        "answer": "B",
        "explanation": "because",
        "options": ("a", "b", "c"),
    }
    base.update(kw)
    return quiz.Question(**base)  # type: ignore[arg-type]


# ------------------------------------------------------------------ banks


def test_every_lesson_has_a_bank_and_every_bank_a_lesson() -> None:
    lessons = {p.name[:2] for p in LEARN.glob("[0-9][0-9]-*.md")}
    assert lessons == set(BANKS), lessons ^ set(BANKS)


def test_question_ids_are_unique_and_prefixed_by_module() -> None:
    ids = [x.id for x in ALL]
    assert len(ids) == len(set(ids))
    for x in ALL:
        assert re.fullmatch(rf"q{x.module}-\d\d", x.id), x.id


def test_banks_are_a_reasonable_size_and_mix() -> None:
    for bank in BANKS.values():
        assert 10 <= len(bank.questions) <= 25, bank.module
        kinds = {x.type for x in bank.questions}
        assert len(kinds) >= 2, bank.module


def _namespace() -> dict[str, object]:

    return {"math": math, "np": np, "stats": scipy.stats, "fplh": sys.modules["fplh"]}


@pytest.mark.parametrize("question", [x for x in ALL if x.check], ids=lambda x: x.id)
def test_numeric_answers_are_reproducible(question: quiz.Question) -> None:
    value = eval(question.check, _namespace())  # the bank's own recomputation
    assert quiz.grade(question, repr(float(value))) == 1.0, (question.answer, value)


def test_numeric_questions_have_a_check_or_are_integers() -> None:
    for x in ALL:
        if x.type == "numeric" and not x.check:
            assert float(x.answer).is_integer(), f"{x.id}: add a check expression"


def test_free_questions_have_rubrics() -> None:
    for x in ALL:
        if x.type == "free":
            assert x.rubric, x.id


def test_lesson_links_resolve() -> None:
    for md in LEARN.glob("*.md"):
        for target in re.findall(r"\]\(([^)#]+)(?:#[^)]*)?\)", md.read_text()):
            if target.startswith("http"):
                continue
            assert (md.parent / target).exists(), f"{md.name}: {target}"


def test_code_references_point_at_real_lines() -> None:
    root = LEARN.parents[1]
    for md in LEARN.glob("*.md"):
        for path, line in re.findall(r"`(src/fplh/[\w/]+\.py):(\d+)", md.read_text()):
            f = root / path
            assert f.exists(), f"{md.name}: {path}"
            assert int(line) <= len(f.read_text().splitlines()), f"{md.name}: {path}:{line}"


# ------------------------------------------------------------------ grading


def test_mcq_grading() -> None:
    x = q()
    assert quiz.grade(x, "b") == 1.0
    assert quiz.grade(x, " B ") == 1.0
    assert quiz.grade(x, "A") == 0.0
    assert quiz.grade(x, "AB") == 0.0


def test_multi_partial_credit() -> None:
    x = q(type="multi", answer=["A", "C"], options=("a", "b", "c", "d"))
    assert quiz.grade(x, "AC") == 1.0
    assert quiz.grade(x, "ca") == 1.0
    assert quiz.grade(x, "A") == 0.5
    assert quiz.grade(x, "AB") == 0.0  # one hit, one false pick
    assert quiz.grade(x, "ABCD") == 0.0


def test_numeric_parsing_and_tolerance() -> None:
    x = q(type="numeric", answer=0.25, tolerance=0.001, options=())
    for text in ("0.25", "25%", "1/4", ".2505", " 0.2495 "):
        assert quiz.grade(x, text) == 1.0, text
    assert quiz.grade(x, "0.26") == 0.0
    assert quiz.grade(x, "abc") == 0.0
    rel = q(type="numeric", answer=1000.0, rel_tolerance=0.01, options=())
    assert quiz.grade(rel, "1009") == 1.0
    assert quiz.grade(rel, "1011") == 0.0


def test_free_self_grade() -> None:
    x = q(type="free", answer="model", options=(), rubric=("r",))
    assert [quiz.grade(x, s) for s in ("0", "1", "2", "7", "x")] == [0.0, 0.5, 1.0, 0.0, 0.0]


def test_bank_validation_rejects_bad_questions() -> None:
    with pytest.raises(quiz.BankError):
        quiz._question(
            "01", {"id": "q01-01", "type": "essay", "prompt": "", "answer": 1, "explanation": ""}
        )
    with pytest.raises(quiz.BankError):
        quiz._question(
            "01",
            {
                "id": "q01-01",
                "type": "mcq",
                "prompt": "?",
                "answer": "E",
                "options": ["a", "b"],
                "explanation": ".",
            },
        )
    with pytest.raises(quiz.BankError):
        quiz._question(
            "01",
            {"id": "q01-01", "type": "numeric", "prompt": "?", "answer": 1.0, "explanation": "."},
        )


def test_weighted_score_uses_difficulty() -> None:
    easy, hard = q(difficulty=1), q(id="qx-02", difficulty=3)
    pts, mx = quiz.weighted_score([(easy, 1.0), (hard, 0.0)])
    assert (pts, mx) == (1.0, 4.0)
    assert quiz.percent(pts, mx) == 25.0


# ------------------------------------------------------------------ progress and report


def test_end_to_end_session(tmp_path: Path) -> None:
    scores, hist = tmp_path / "scores.csv", tmp_path / "questions.json"
    bank = BANKS["00"]
    qs = list(bank.questions)[:3]
    answers = iter([str(x.answer) if x.type != "free" else "2" for x in qs])
    said: list[str] = []
    per_q = quiz.run_session(qs, lambda _p: next(answers), said.append)
    assert all(v == 1.0 for v in per_q.values()), per_q
    pts, mx = quiz.weighted_score((x, per_q[x.id]) for x in qs)
    quiz.append_score("00", "cli", pts, mx, per_q, path=scores)
    quiz.update_history(per_q, path=hist)
    rows = quiz.read_scores(scores)
    assert rows[0]["module"] == "00"
    assert float(rows[0]["score"]) == pts
    status = {s.module: s for s in quiz.module_status(BANKS, rows, {})}
    assert status["00"].best == 100.0
    assert status["00"].mastered
    text = quiz.render_report(BANKS, rows, quiz.load_history(hist), {})
    assert "next: module 01" in text


def test_mastery_needs_the_practical(tmp_path: Path) -> None:
    rows = [
        {"module": "02", "mode": "cli", "score": "9", "max": "10"},
        {"module": "02", "mode": "practical", "score": "0", "max": "1"},
    ]
    prac = {"02": tmp_path / "ex02.py"}
    st = {s.module: s for s in quiz.module_status(BANKS, rows, prac)}["02"]
    assert st.best == 90.0
    assert st.practical is False
    assert not st.mastered
    rows.append({"module": "02", "mode": "practical", "score": "1", "max": "1"})
    st = {s.module: s for s in quiz.module_status(BANKS, rows, prac)}["02"]
    assert st.mastered


def test_time_limit_scores_the_rest_zero() -> None:
    qs = list(BANKS["00"].questions)[:4]
    ticks = iter([0.0, 0.0, 10.0, 999.0, 999.0, 999.0])
    per_q = quiz.run_session(
        qs, lambda _p: "A", lambda _s: None, time_limit_s=60, clock=lambda: next(ticks)
    )
    assert list(per_q) == [x.id for x in qs]
    assert all(per_q[x.id] == 0.0 for x in qs[2:])


def test_leitner_boxes_and_review_order(tmp_path: Path) -> None:
    hist_path = tmp_path / "q.json"
    quiz.update_history({"q00-01": 1.0, "q00-02": 0.0}, path=hist_path)
    quiz.update_history({"q00-01": 1.0}, path=hist_path)
    h = quiz.load_history(hist_path)
    assert h["q00-01"]["box"] == 3
    assert h["q00-02"]["box"] == 1
    qs = [x for x in BANKS["00"].questions if x.id in ("q00-01", "q00-02", "q00-03")]
    order = [x.id for x in quiz.review_order(qs, h, random.Random(0))]
    assert order[-1] == "q00-01"  # the well-known one comes last


def test_course_score_weights() -> None:
    s = quiz.ModuleStatus("01", "t", 10, best=100.0, practical=True, has_practical=True)
    t = quiz.ModuleStatus("02", "t", 10, best=50.0, practical=False, has_practical=True)
    assert quiz.course_score([s, t]) == pytest.approx(0.7 * 75 + 0.3 * 50)
