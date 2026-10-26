import json

import pytest
from conftest import QUESTIONS, FakeClient

from captionexam import board, caption, judge
from captionexam.data import (
    NOT_MENTIONED,
    append_jsonl,
    caption_fp,
    check_question,
    question_fp,
    read_jsonl,
)
from captionexam.judge import verdict_mode


def test_check_question_catches_bad_rows(bench):
    good = QUESTIONS[0]
    assert check_question(good, bench.images) == []
    assert check_question({**good, "options": ["a", "b"]}, bench.images)
    assert check_question({**good, "answer": [NOT_MENTIONED]}, bench.images)
    assert check_question({**good, "bbox": [500, 0, 100, 1000]}, bench.images)
    assert check_question({**good, "image_id": "nope"}, bench.images)


def test_fingerprints():
    q = QUESTIONS[0]
    assert question_fp(q) == question_fp(dict(q))
    assert question_fp(q) != question_fp({**q, "answer": ["black"]})
    assert caption_fp({"a": 1}) == caption_fp('{"a": 1}')


def test_verdict_mode():
    assert verdict_mode(["orange"], ["orange"]) == "correct"
    assert verdict_mode([NOT_MENTIONED], ["orange"]) == "omission"
    assert verdict_mode(["black"], ["orange"]) == "hallucination"


def test_read_jsonl_skips_truncated_line(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text('{"a": 1}\n{"a": 2')
    assert read_jsonl(p) == [{"a": 1}]


def judge_reply(answers):
    """A fake judge that answers each question with answers[question text]."""

    def reply(prompt):
        for text, choice in answers.items():
            if text in prompt:
                return json.dumps({"choice": [choice]})
        raise AssertionError("unexpected prompt")

    return reply


@pytest.fixture
def results(tmp_path, bench, monkeypatch):
    out = tmp_path / "results"
    fake = FakeClient(lambda prompt: "An orange cat next to a mug that says HELLO.")
    monkeypatch.setattr(caption, "make_client", lambda *a, **k: fake)
    caption.run_captions(bench, out / "captions" / "m1.jsonl", "m1-hf", "m1", "Describe.")
    return out


def test_caption_skips_done_images(bench, results, monkeypatch):
    rows = read_jsonl(results / "captions" / "m1.jsonl")
    assert sorted(r["id"] for r in rows) == ["img_0001", "img_0002"]
    fake = FakeClient(lambda prompt: "unused")
    monkeypatch.setattr(caption, "make_client", lambda *a, **k: fake)
    caption.run_captions(bench, results / "captions" / "m1.jsonl", "m1-hf", "m1", "Describe.")
    assert fake.calls == []
    # A new prompt redoes every image
    caption.run_captions(bench, results / "captions" / "m1.jsonl", "m1-hf", "m1", "Other.")
    assert len(fake.calls) == 2


def test_judge_scores_and_reruns_only_gaps(bench, results, monkeypatch):
    answers = {
        "What color is the cat?": "orange",
        "What is written on the mug?": "HELL0",
        "How many chairs are there?": NOT_MENTIONED,
    }
    fake = FakeClient(judge_reply(answers))
    monkeypatch.setattr(judge, "make_client", lambda *a, **k: fake)
    caps, ledger = results / "captions" / "m1.jsonl", results / "runs" / "m1.jsonl"
    judge.run_judge(bench, caps, ledger, judge_model="phi")
    modes = {r["question"]: r["mode"] for r in read_jsonl(ledger)}
    assert modes == {
        "img_0001_q01": "correct",
        "img_0001_q02": "hallucination",
        "img_0002_q01": "omission",
    }
    # The answer is constrained to the options' exact text
    schema = fake.calls[0]["response_format"]["json_schema"]["schema"]
    assert NOT_MENTIONED in schema["properties"]["choice"]["items"]["enum"]

    judge.run_judge(bench, caps, ledger, judge_model="phi")
    assert len(fake.calls) == 3  # nothing left to judge

    # A regenerated caption makes that image's verdicts stale
    append_jsonl(caps, [{"id": "img_0002", "caption": "Three chairs.", "prompt_fp": "x"}])
    answers["How many chairs are there?"] = "three"
    judge.run_judge(bench, caps, ledger, judge_model="phi")
    assert len(fake.calls) == 4

    b = board.compute(bench, results, judge="phi")
    (row,) = b["rows"]
    assert row["n"] == 3 and row["n_correct"] == 2 and row["n_hallucination"] == 1
    assert "| 1 | m1 | 66.7 |" in board.render(b)


def test_board_ranks_on_common_questions(bench, results, monkeypatch):
    caps = results / "captions"
    (caps / "m2.jsonl").write_text(json.dumps({"id": "img_0001", "caption": "A cat."}) + "\n")
    fake = FakeClient(
        judge_reply(
            {
                "What color is the cat?": "orange",
                "What is written on the mug?": "HELLO",
                "How many chairs are there?": "three",
            }
        )
    )
    monkeypatch.setattr(judge, "make_client", lambda *a, **k: fake)
    for m in ("m1", "m2"):
        judge.run_judge(bench, caps / f"{m}.jsonl", results / "runs" / f"{m}.jsonl", "phi")
    b = board.compute(bench, results, judge="phi")
    assert b["common"] == ["img_0001_q01", "img_0001_q02"]  # m2 has no caption for img_0002
    assert {r["model"]: r["own"]["n"] for r in b["rows"]} == {"m1": 3, "m2": 2}
