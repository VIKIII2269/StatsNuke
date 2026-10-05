"""Quiz runner and score tracker for the StatsNuke course (docs/learn).

Run from the repository root:

    uv run python docs/learn/quiz.py list              # modules, question counts, mastery
    uv run python docs/learn/quiz.py take 04           # one module's quiz
    uv run python docs/learn/quiz.py review            # spaced repetition over everything
    uv run python docs/learn/quiz.py exam              # the mixed final exam
    uv run python docs/learn/quiz.py practicals        # run the exercise scripts, record pass/fail
    uv run python docs/learn/quiz.py report            # scores, mastery, weakest tags, next module
    uv run python docs/learn/quiz.py record 04 q04-01=1 q04-02=0.5   # log a quiz graded in chat

Scores are appended to ``progress/scores.csv``; per-question history (for spaced repetition) is in
``progress/questions.json``. Commit both to keep your progress.
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from typing import Annotated, Any

import typer
import yaml

HERE = Path(__file__).resolve().parent
BANKS = HERE / "quizzes"
EXERCISES = HERE / "exercises"
PROGRESS = HERE / "progress"
SCORES_CSV = PROGRESS / "scores.csv"
QUESTIONS_JSON = PROGRESS / "questions.json"

MASTERY = 80.0  # quiz score (best attempt, %) a module needs to count as mastered
QUIZ_WEIGHT, PRACTICAL_WEIGHT = 0.7, 0.3
LETTERS = "ABCDEFGHIJ"
TYPES = {"mcq", "multi", "numeric", "free"}
FIELDS = ["timestamp", "module", "mode", "score", "max", "per_question_json"]
MAX_BOX = 5


# ---------------------------------------------------------------- banks


@dataclass(frozen=True)
class Question:
    id: str
    module: str
    type: str
    prompt: str
    answer: Any
    explanation: str
    difficulty: int = 1
    tags: tuple[str, ...] = ()
    options: tuple[str, ...] = ()
    tolerance: float = 0.0
    rel_tolerance: float = 0.0
    rubric: tuple[str, ...] = ()
    check: str | None = None


@dataclass(frozen=True)
class Bank:
    module: str
    title: str
    path: Path
    questions: tuple[Question, ...]


class BankError(ValueError):
    pass


def _question(module: str, raw: dict[str, Any]) -> Question:
    qid = str(raw.get("id", "?"))
    kind = raw.get("type")
    if kind not in TYPES:
        raise BankError(f"{qid}: type must be one of {sorted(TYPES)}, got {kind!r}")
    for key in ("id", "prompt", "answer", "explanation"):
        if key not in raw:
            raise BankError(f"{qid}: missing {key!r}")
    difficulty = int(raw.get("difficulty", 1))
    if difficulty not in (1, 2, 3):
        raise BankError(f"{qid}: difficulty must be 1, 2 or 3")
    options = tuple(str(o) for o in raw.get("options", ()))
    answer = raw["answer"]
    if kind in ("mcq", "multi"):
        if len(options) < 2:
            raise BankError(f"{qid}: {kind} needs at least two options")
        picks = [answer] if kind == "mcq" else answer
        if not isinstance(picks, list) or not picks:
            raise BankError(f"{qid}: multi answer must be a non-empty list of letters")
        for letter in picks:
            if not isinstance(letter, str) or letter not in LETTERS[: len(options)]:
                raise BankError(f"{qid}: answer {letter!r} is not an option letter")
    if kind == "numeric":
        if not isinstance(answer, int | float):
            raise BankError(f"{qid}: numeric answer must be a number")
        if "tolerance" not in raw and "rel_tolerance" not in raw:
            raise BankError(f"{qid}: numeric needs tolerance or rel_tolerance")
    return Question(
        id=qid,
        module=module,
        type=kind,
        prompt=str(raw["prompt"]).strip(),
        answer=answer,
        explanation=str(raw["explanation"]).strip(),
        difficulty=difficulty,
        tags=tuple(raw.get("tags", ())),
        options=options,
        tolerance=float(raw.get("tolerance", 0.0)),
        rel_tolerance=float(raw.get("rel_tolerance", 0.0)),
        rubric=tuple(str(r) for r in raw.get("rubric", ())),
        check=raw.get("check"),
    )


def load_bank(path: Path) -> Bank:
    raw = yaml.safe_load(path.read_text())
    module = str(raw["module"]).zfill(2)
    questions = tuple(_question(module, q) for q in raw["questions"])
    ids = [q.id for q in questions]
    if len(set(ids)) != len(ids):
        raise BankError(f"{path.name}: duplicate question ids")
    return Bank(module=module, title=str(raw["title"]), path=path, questions=questions)


def load_banks(directory: Path = BANKS) -> dict[str, Bank]:
    banks = [load_bank(p) for p in sorted(directory.glob("*.yaml"))]
    return {b.module: b for b in banks}


def find_question(banks: dict[str, Bank], qid: str) -> Question:
    for bank in banks.values():
        for q in bank.questions:
            if q.id == qid:
                return q
    raise KeyError(qid)


# ---------------------------------------------------------------- grading


def parse_number(text: str) -> float:
    """Parse '0.25', '25%', '1/4', '-1.5e-3' or '2,500' into a float."""
    s = text.strip().replace(",", "").replace("−", "-")
    if not s:
        raise ValueError("empty answer")
    if s.endswith("%"):
        return float(s[:-1]) / 100
    if "/" in s:
        return float(Fraction(s))
    return float(s)


def parse_letters(text: str) -> set[str]:
    return {c for c in text.upper() if c in LETTERS}


def grade(q: Question, response: str) -> float:
    """Credit in [0, 1] for a typed response. Free-text questions take a self-grade 0/1/2."""
    if q.type == "mcq":
        picks = parse_letters(response)
        return 1.0 if picks == {q.answer} else 0.0
    if q.type == "multi":
        picks = parse_letters(response)
        right = set(q.answer)
        hits, false_picks = len(picks & right), len(picks - right)
        return max(0.0, (hits - false_picks) / len(right))
    if q.type == "numeric":
        try:
            value = parse_number(response)
        except (ValueError, ZeroDivisionError):
            return 0.0
        target = float(q.answer)
        allowed = max(q.tolerance, q.rel_tolerance * abs(target))
        return 1.0 if math.isclose(value, target, abs_tol=allowed + 1e-12) else 0.0
    # free: response is the self-grade
    try:
        points = int(response.strip())
    except ValueError:
        return 0.0
    return {0: 0.0, 1: 0.5, 2: 1.0}.get(points, 0.0)


def correct_answer(q: Question) -> str:
    if q.type == "mcq":
        return f"{q.answer}) {q.options[LETTERS.index(q.answer)]}"
    if q.type == "multi":
        return ", ".join(f"{a}) {q.options[LETTERS.index(a)]}" for a in q.answer)
    if q.type == "numeric":
        tol = max(q.tolerance, q.rel_tolerance * abs(float(q.answer)))
        return f"{q.answer} (± {tol:g})"
    return str(q.answer)


def weighted_score(results: Iterable[tuple[Question, float]]) -> tuple[float, float]:
    """(points, max points) where each question is worth its difficulty."""
    pts = mx = 0.0
    for q, credit in results:
        pts += q.difficulty * credit
        mx += q.difficulty
    return pts, mx


def percent(points: float, maximum: float) -> float:
    return 100.0 * points / maximum if maximum else 0.0


# ---------------------------------------------------------------- progress


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def append_score(
    module: str,
    mode: str,
    points: float,
    maximum: float,
    per_question: dict[str, float],
    path: Path = SCORES_CSV,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(FIELDS)
        w.writerow(
            [_now(), module, mode, f"{points:.3f}", f"{maximum:.3f}", json.dumps(per_question)]
        )


def read_scores(path: Path = SCORES_CSV) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def load_history(path: Path = QUESTIONS_JSON) -> dict[str, dict[str, Any]]:
    if not path.exists() or not path.read_text().strip():
        return {}
    data: dict[str, dict[str, Any]] = json.loads(path.read_text())
    return data


def update_history(
    per_question: dict[str, float], path: Path = QUESTIONS_JSON
) -> dict[str, dict[str, Any]]:
    """Leitner boxes: a full-credit answer moves a question up a box, anything else back to 1."""
    hist = load_history(path)
    for qid, credit in per_question.items():
        h = hist.setdefault(qid, {"box": 1, "seen": 0, "credit_sum": 0.0, "last": ""})
        h["seen"] += 1
        h["credit_sum"] = round(h["credit_sum"] + credit, 3)
        h["box"] = min(MAX_BOX, h["box"] + 1) if credit >= 0.999 else 1
        h["last"] = _now()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(hist, indent=1, sort_keys=True) + "\n")
    return hist


def review_order(
    questions: list[Question], history: dict[str, dict[str, Any]], rng: random.Random
) -> list[Question]:
    """Unseen and low-box questions first; within a box, the longest-unseen first."""

    def key(q: Question) -> tuple[int, str, float]:
        h = history.get(q.id)
        if h is None:
            return (1, "", rng.random())  # unseen ranks with box 1, before seen box-1 items
        return (int(h["box"]), str(h["last"]), rng.random())

    return sorted(questions, key=key)


# ---------------------------------------------------------------- report


@dataclass
class ModuleStatus:
    module: str
    title: str
    n_questions: int
    attempts: int = 0
    latest: float | None = None
    best: float | None = None
    practical: bool | None = None  # None: no exercise for this module, or never run
    has_practical: bool = False
    weak_tags: list[str] = field(default_factory=list)

    @property
    def mastered(self) -> bool:
        quiz_ok = self.best is not None and self.best >= MASTERY
        return quiz_ok and (self.practical is True or not self.has_practical)


def practical_modules(directory: Path = EXERCISES) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for p in sorted(directory.glob("ex[0-9][0-9]_*.py")):
        out[p.name[2:4]] = p
    return out


def module_status(
    banks: dict[str, Bank],
    scores: list[dict[str, str]],
    practicals: dict[str, Path],
) -> list[ModuleStatus]:
    out = []
    for module, bank in sorted(banks.items()):
        st = ModuleStatus(
            module, bank.title, len(bank.questions), has_practical=module in practicals
        )
        for row in scores:
            if row["module"] != module:
                continue
            if row["mode"] == "practical":
                st.practical = float(row["score"]) >= float(row["max"]) > 0
                continue
            pct = percent(float(row["score"]), float(row["max"]))
            st.attempts += 1
            st.latest = pct
            st.best = pct if st.best is None else max(st.best, pct)
        out.append(st)
    return out


def tag_accuracy(
    banks: dict[str, Bank], history: dict[str, dict[str, Any]]
) -> dict[str, tuple[float, int]]:
    """Mean credit per tag over every answered question: tag -> (accuracy, answers)."""
    sums: dict[str, list[float]] = {}
    for bank in banks.values():
        for q in bank.questions:
            h = history.get(q.id)
            if not h or not h["seen"]:
                continue
            for tag in (*q.tags, f"module-{q.module}"):
                s = sums.setdefault(tag, [0.0, 0.0])
                s[0] += h["credit_sum"]
                s[1] += h["seen"]
    return {t: (s[0] / s[1], int(s[1])) for t, s in sums.items()}


def course_score(statuses: list[ModuleStatus]) -> float:
    if not statuses:
        return 0.0
    quiz = sum(s.best or 0.0 for s in statuses) / len(statuses)
    with_prac = [s for s in statuses if s.has_practical]
    prac = 100.0 * sum(s.practical is True for s in with_prac) / len(with_prac) if with_prac else 0
    return QUIZ_WEIGHT * quiz + PRACTICAL_WEIGHT * prac


def next_module(statuses: list[ModuleStatus]) -> ModuleStatus | None:
    return next((s for s in statuses if not s.mastered), None)


def render_report(
    banks: dict[str, Bank],
    scores: list[dict[str, str]],
    history: dict[str, dict[str, Any]],
    practicals: dict[str, Path],
) -> str:
    statuses = module_status(banks, scores, practicals)
    lines = [
        f"{'mod':<4}{'title':<36}{'tries':>6}{'latest':>8}{'best':>7}  {'practical':<10}mastered",
        "-" * 82,
    ]
    for s in statuses:
        latest = "-" if s.latest is None else f"{s.latest:.0f}%"
        best = "-" if s.best is None else f"{s.best:.0f}%"
        prac = (
            "n/a" if not s.has_practical else {True: "pass", False: "FAIL", None: "-"}[s.practical]
        )
        tick = "yes" if s.mastered else ""
        lines.append(
            f"{s.module:<4}{s.title[:35]:<36}{s.attempts:>6}{latest:>8}{best:>7}  {prac:<10}{tick}"
        )
    exams = [r for r in scores if r["module"] == "exam"]
    lines.append("-" * 82)
    if exams:
        last = exams[-1]
        best_exam = max(percent(float(r["score"]), float(r["max"])) for r in exams)
        lines.append(
            f"final exam: {len(exams)} attempt(s), latest "
            f"{percent(float(last['score']), float(last['max'])):.0f}%, best {best_exam:.0f}%"
        )
    mastered = sum(s.mastered for s in statuses)
    lines.append(
        f"mastered {mastered}/{len(statuses)} modules · course score "
        f"{course_score(statuses):.1f}/100 (quizzes {QUIZ_WEIGHT:.0%}, "
        f"practicals {PRACTICAL_WEIGHT:.0%}) · mastery bar {MASTERY:.0f}%"
    )
    acc = tag_accuracy(banks, history)
    topical = {t: v for t, v in acc.items() if not t.startswith("module-") and v[1] >= 2}
    if topical:
        weakest = sorted(topical.items(), key=lambda kv: kv[1][0])[:5]
        lines.append(
            "weakest tags: " + ", ".join(f"{t} {a:.0%} ({n} answers)" for t, (a, n) in weakest)
        )
    due = sum(1 for h in history.values() if h["box"] <= 2)
    if history:
        lines.append(f"review queue: {due} question(s) in boxes 1–2 → `quiz.py review`")
    nxt = next_module(statuses)
    lines.append(
        "next: all modules mastered — take `quiz.py exam`"
        if nxt is None
        else f"next: module {nxt.module} — {nxt.title}"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------- interactive session

Ask = Callable[[str], str]
Say = Callable[[str], None]


def ask_question(q: Question, n: int, total: int, ask: Ask, say: Say) -> float:
    say(f"\n[{n}/{total}] {q.id} · {q.type} · difficulty {q.difficulty} · {', '.join(q.tags)}")
    say(q.prompt)
    for letter, opt in zip(LETTERS, q.options, strict=False):
        say(f"  {letter}) {opt}")
    hints = {
        "mcq": "one letter",
        "multi": "every correct letter, e.g. AC",
        "numeric": "a number (0.25, 25%, 1/4 all work)",
        "free": "write your answer, then grade yourself",
    }
    response = ask(f"answer ({hints[q.type]}): ")
    if q.type == "free":
        say("model answer: " + str(q.answer).strip())
        for point in q.rubric:
            say(f"  · {point}")
        response = ask("self-grade — 0 missed it, 1 partly, 2 fully: ")
        credit = grade(q, response)
    else:
        credit = grade(q, response)
        mark = "correct" if credit >= 0.999 else ("partly right" if credit > 0 else "wrong")
        say(f"{mark} — answer: {correct_answer(q)}")
    say("why: " + q.explanation)
    return credit


def run_session(
    questions: list[Question],
    ask: Ask,
    say: Say,
    time_limit_s: float | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, float]:
    start = clock()
    per_question: dict[str, float] = {}
    for i, q in enumerate(questions, 1):
        if time_limit_s is not None and clock() - start > time_limit_s:
            say(f"\ntime is up — {len(questions) - i + 1} unanswered question(s) score 0")
            for rest in questions[i - 1 :]:
                per_question[rest.id] = 0.0
            break
        per_question[q.id] = ask_question(q, i, len(questions), ask, say)
    return per_question


def finish(
    module: str, mode: str, questions: list[Question], per_question: dict[str, float], say: Say
) -> float:
    by_id = {q.id: q for q in questions}
    pts, mx = weighted_score((by_id[k], v) for k, v in per_question.items())
    pct = percent(pts, mx)
    append_score(module, mode, pts, mx, per_question)
    update_history(per_question)
    verdict = "MASTERED" if pct >= MASTERY else f"below the {MASTERY:.0f}% mastery bar"
    say(f"\nscore: {pts:g}/{mx:g} = {pct:.0f}% — {verdict}")
    missed = [k for k, v in per_question.items() if v < 0.999]
    if missed:
        say("revisit: " + ", ".join(missed))
    return pct


def _input(prompt: str) -> str:
    try:
        return input(prompt)
    except EOFError:
        return ""


# ---------------------------------------------------------------- CLI

app = typer.Typer(no_args_is_help=True, help="StatsNuke course: quizzes, practicals and scores.")


def _resolve(banks: dict[str, Bank], module: str) -> Bank:
    key = module.zfill(2)
    if key not in banks:
        raise typer.BadParameter(f"no quiz bank for module {module!r}; try `list`")
    return banks[key]


@app.command("list")
def list_cmd() -> None:
    """Modules, question counts and mastery."""
    banks = load_banks()
    statuses = module_status(banks, read_scores(), practical_modules())
    for s in statuses:
        mark = "✓" if s.mastered else " "
        best = "" if s.best is None else f" best {s.best:.0f}%"
        typer.echo(f"{mark} {s.module} {s.title} ({s.n_questions} questions){best}")


@app.command()
def take(
    module: str,
    n: Annotated[int, typer.Option(help="Ask at most this many questions (0 = all).")] = 0,
    seed: Annotated[int | None, typer.Option(help="Fix the question order.")] = None,
) -> None:
    """Take one module's quiz; the score is recorded."""
    bank = _resolve(load_banks(), module)
    qs = list(bank.questions)
    random.Random(seed).shuffle(qs)
    if n:
        qs = qs[:n]
    typer.echo(f"Module {bank.module}: {bank.title} — {len(qs)} questions")
    per_q = run_session(qs, _input, typer.echo)
    finish(bank.module, "cli", qs, per_q, typer.echo)


@app.command()
def review(
    n: Annotated[int, typer.Option(help="Number of questions.")] = 10,
    module: Annotated[str | None, typer.Option(help="Only this module.")] = None,
    seed: Annotated[int | None, typer.Option()] = None,
) -> None:
    """Spaced repetition: weakest and least-recent questions first."""
    banks = load_banks()
    pool = [q for b in banks.values() for q in b.questions]
    seen_modules = {k.split("-")[0][1:] for k in load_history()}
    if module:
        pool = [q for q in pool if q.module == module.zfill(2)]
    elif seen_modules:
        pool = [q for q in pool if q.module in seen_modules]  # review what you've studied
    qs = review_order(pool, load_history(), random.Random(seed))[:n]
    per_q = run_session(qs, _input, typer.echo)
    by_id = {q.id: q for q in qs}
    pts, mx = weighted_score((by_id[k], v) for k, v in per_q.items())
    update_history(per_q)
    append_score("review", "review", pts, mx, per_q)
    typer.echo(f"\nreview score: {percent(pts, mx):.0f}% (does not change module bests)")


@app.command()
def exam(
    per_module: Annotated[int, typer.Option(help="Questions drawn from each module.")] = 2,
    minutes: Annotated[float, typer.Option(help="Time limit.")] = 60.0,
    seed: Annotated[int | None, typer.Option()] = None,
) -> None:
    """The final exam: questions from every module, timed."""
    banks = load_banks()
    rng = random.Random(seed)
    qs: list[Question] = []
    for bank in banks.values():
        pool = list(bank.questions)
        rng.shuffle(pool)
        qs.extend(pool[:per_module])
    rng.shuffle(qs)
    typer.echo(f"Final exam: {len(qs)} questions, {minutes:g} minutes")
    per_q = run_session(qs, _input, typer.echo, time_limit_s=minutes * 60)
    finish("exam", "exam", qs, per_q, typer.echo)


@app.command()
def practicals(
    module: Annotated[str | None, typer.Option(help="Only this module's exercise.")] = None,
) -> None:
    """Run the exercise scripts; each passes when every task in it is solved."""
    scripts = practical_modules()
    if module:
        scripts = {k: v for k, v in scripts.items() if k == module.zfill(2)}
    for mod, path in scripts.items():
        env = {k: v for k, v in os.environ.items() if k != "LEARN_SOLUTIONS"}
        proc = subprocess.run(
            [sys.executable, str(path)], capture_output=True, text=True, env=env, check=False
        )
        ok = proc.returncode == 0
        summary = re.findall(r"^(?:PASS|FAIL|TODO) .*$", proc.stdout, re.MULTILINE)
        append_score(mod, "practical", 1.0 if ok else 0.0, 1.0, {})
        typer.echo(f"{'pass' if ok else 'FAIL'}  {path.name}")
        for line in summary:
            typer.echo(f"      {line}")
        if proc.returncode not in (0, 1):
            typer.echo(proc.stderr.strip()[-800:])


@app.command()
def report() -> None:
    """Scores per module, mastery, weakest tags and what to study next."""
    typer.echo(render_report(load_banks(), read_scores(), load_history(), practical_modules()))


@app.command()
def record(
    module: str,
    grades: Annotated[list[str], typer.Argument(help="question_id=credit, credit in [0, 1]")],
    mode: Annotated[str, typer.Option(help="chat, cli or exam.")] = "chat",
) -> None:
    """Log a quiz graded elsewhere (e.g. in chat with Claude)."""
    banks = load_banks()
    per_q: dict[str, float] = {}
    for item in grades:
        qid, _, value = item.partition("=")
        find_question(banks, qid)  # raises on a typo
        credit = float(value)
        if not 0 <= credit <= 1:
            raise typer.BadParameter(f"{item}: credit must be in [0, 1]")
        per_q[qid] = credit
    qs = [find_question(banks, k) for k in per_q]
    key = "exam" if module == "exam" else ("diagnostic" if module == "diagnostic" else None)
    target = key or _resolve(banks, module).module
    finish(target, mode, qs, per_q, typer.echo)


if __name__ == "__main__":
    app()
