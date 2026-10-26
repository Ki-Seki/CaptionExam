"""The benchmark files and the JSONL records written while evaluating.

A benchmark directory holds `images/<image_id>.jpg` and `questions.jsonl`, one question per
line: {"id", "image_id", "question", "options", "answer", "bbox"}. Every question has exactly
one NOT_MENTIONED option, which is never the answer.
"""

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

DATASET_REPO = "Ki-Seki/CaptionExam"
NOT_MENTIONED = "not mentioned / indeterminable"


@dataclass
class Benchmark:
    root: Path
    images: dict  # image_id -> image path
    questions: list  # question dicts, in file order

    def by_image(self):
        out = {}
        for q in self.questions:
            out.setdefault(q["image_id"], []).append(q)
        return out


def load_benchmark(root):
    """Load a benchmark directory and check every question against the format above."""
    root = Path(root)
    if not (root / "questions.jsonl").exists():
        raise SystemExit(
            f"No questions.jsonl in {root}. Run `captionexam download` first, or pass --data."
        )
    images = {
        p.stem: p
        for p in sorted((root / "images").iterdir())
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
    }
    questions = read_jsonl(root / "questions.jsonl")
    for q in questions:
        problems = check_question(q, images)
        if problems:
            raise SystemExit(f"Invalid question {q.get('id')}: {'; '.join(problems)}")
    return Benchmark(root, images, questions)


def check_question(q, images):
    problems = []
    if q.get("image_id") not in images:
        problems.append("unknown image_id")
    opts, ans = q.get("options") or [], q.get("answer") or []
    if not q.get("question"):
        problems.append("empty question")
    if opts.count(NOT_MENTIONED) != 1:
        problems.append(f"needs exactly one {NOT_MENTIONED!r} option")
    if not ans or not set(ans) <= set(opts) or NOT_MENTIONED in ans:
        problems.append("answer must be a non-empty subset of the other options")
    b = q.get("bbox")
    if not (
        isinstance(b, list)
        and len(b) == 4
        and all(0 <= v <= 1000 for v in b)
        and b[0] < b[2]
        and b[1] < b[3]
    ):
        problems.append("bbox must be [y_min, x_min, y_max, x_max] within 0-1000")
    return problems


def download(dest, repo_id=DATASET_REPO, revision=None):
    """Fetch the benchmark from the Hugging Face Hub into `dest`."""
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id, repo_type="dataset", local_dir=dest, revision=revision))


def question_fp(q):
    """Fingerprint of everything the judge sees about a question.

    A verdict recorded under another fingerprint is stale: the question changed since.
    """
    blob = json.dumps([q["question"], q["options"], q["answer"]], sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def caption_text(caption):
    """Captions may be plain text or structured (JSON); the judge reads them as text."""
    if isinstance(caption, str):
        return caption
    return json.dumps(caption, ensure_ascii=False)


def caption_fp(caption):
    return hashlib.sha1(caption_text(caption).encode()).hexdigest()[:12]


def read_jsonl(path):
    """Read a JSONL file, skipping blank lines and a truncated last line."""
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a process killed mid-write leaves half a line
    return rows


def append_jsonl(path, rows):
    """Append rows in one write, so an interrupted run never leaves a partial batch."""
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    with path.open("a") as f:
        f.write(blob)
        f.flush()
        os.fsync(f.fileno())


def latest_by(rows, key):
    """Fold an append-only log: for each key, the last row wins."""
    out = {}
    for r in rows:
        out[r[key]] = r
    return out
