"""Grade captions: a text-only judge answers every question from the caption alone.

Each verdict is one of
  correct        the judge picked the answer
  omission       the judge picked NOT_MENTIONED: the caption does not say
  hallucination  the judge picked a wrong option: the caption says something false
"""

import concurrent.futures as cf
import json
import threading
from datetime import datetime, timezone

from .client import chat, make_client, usage_of
from .data import (
    NOT_MENTIONED,
    append_jsonl,
    caption_fp,
    caption_text,
    latest_by,
    question_fp,
    read_jsonl,
)
from .prompts import judge_prompt

FLUSH_EVERY = 64
# The judge only returns {"choice": [...]}: a few dozen tokens at most.
MAX_TOKENS = 256


def verdict_mode(choice, answer):
    if set(choice) == set(answer):
        return "correct"
    return "omission" if NOT_MENTIONED in choice else "hallucination"


def choice_schema(options):
    """Constrain the answer to the options' exact text, so a paraphrased option can't miss."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "verdict",
            "schema": {
                "type": "object",
                "properties": {
                    "choice": {"type": "array", "items": {"type": "string", "enum": options}}
                },
                "required": ["choice"],
            },
        },
    }


def current_verdicts(ledger, bench, captions, judge):
    """The ledger rows that still apply: same question, same caption, same judge."""
    qfp = {q["id"]: question_fp(q) for q in bench.questions}
    q_image = {q["id"]: q["image_id"] for q in bench.questions}
    cfp = {i: caption_fp(r["caption"]) for i, r in captions.items()}
    out = {}
    for qid, r in latest_by(read_jsonl(ledger), "question").items():
        if (
            qid in qfp
            and r.get("question_fp") == qfp[qid]
            and r.get("caption_fp") == cfp.get(q_image[qid])
            and r.get("judge") == judge
        ):
            out[qid] = r
    return out


def run_judge(
    bench,
    captions_path,
    ledger_path,
    judge_model,
    judge_name=None,
    base_url=None,
    api_key=None,
    workers=32,
    log=print,
):
    """Judge every question whose image has a caption; append verdicts to `ledger_path`.

    The ledger is append-only. Questions with a current verdict are skipped, so a rerun only
    fills gaps, and a changed caption or question is judged again.
    """
    judge = judge_name or judge_model
    captions = latest_by(read_jsonl(captions_path), "id")
    have = current_verdicts(ledger_path, bench, captions, judge)
    jobs = [
        q
        for q in bench.questions
        if q["image_id"] in captions and q["id"] not in have and captions[q["image_id"]]["caption"]
    ]
    no_caption = sum(1 for q in bench.questions if q["image_id"] not in captions)
    log(
        f"{len(have)} verdicts current, {len(jobs)} to judge"
        + (f", {no_caption} questions have no caption" if no_caption else "")
    )
    if not jobs:
        return

    client = make_client(base_url, api_key)
    lock = threading.Lock()
    pending, failed = [], []
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def one(q):
        cap = captions[q["image_id"]]["caption"]
        prompt = judge_prompt(caption_text(cap), q["question"], q["options"])
        r = chat(
            client,
            judge_model,
            prompt,
            max_tokens=MAX_TOKENS,
            temperature=0,
            response_format=choice_schema(q["options"]),
        )
        try:
            choice = json.loads(r.choices[0].message.content).get("choice")
        except (json.JSONDecodeError, TypeError, AttributeError) as e:
            # Unparseable output stays a gap, not an omission, so a rerun retries it
            raise ValueError(f"unparseable judge output: {r.choices[0].message.content!r}") from e
        if isinstance(choice, str):
            choice = [choice]
        choice = [str(c) for c in (choice or [NOT_MENTIONED])]
        u = usage_of(r)
        return {
            "question": q["id"],
            "question_fp": question_fp(q),
            "caption_fp": caption_fp(cap),
            "choice": choice,
            "mode": verdict_mode(choice, q["answer"]),
            "judge": judge,
            "tokens_in": u["prompt_tokens"],
            "tokens_out": u["completion_tokens"],
            "at": stamp,
        }

    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(one, q): q["id"] for q in jobs}
        for n, fut in enumerate(cf.as_completed(futures), 1):
            try:
                row = fut.result()
            except Exception as e:  # noqa: BLE001 - a failed question stays a gap to refill
                failed.append(futures[fut])
                if len(failed) <= 5:
                    log(f"  ! {futures[fut]}: {e}")
                continue
            with lock:
                pending.append(row)
                if len(pending) >= FLUSH_EVERY:
                    append_jsonl(ledger_path, pending)
                    pending.clear()
            if n % 500 == 0 or n == len(jobs):
                log(f"  {n}/{len(jobs)} questions")
    append_jsonl(ledger_path, pending)
    if failed:
        log(f"{len(failed)} questions failed; rerun the same command to retry them")
