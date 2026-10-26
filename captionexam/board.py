"""The leaderboard, computed from the ledgers every time; nothing is cached or stored.

Per model: fold its ledger (the last verdict per question wins), drop stale verdicts, then
score. Models are ranked on the questions they all have current verdicts for, so every row
of the main table is scored on the same questions.
"""

import json
from pathlib import Path

from .data import latest_by, read_jsonl
from .judge import current_verdicts

MODES = ("correct", "omission", "hallucination")


def tally(rows):
    n = len(rows)
    counts = {m: sum(1 for r in rows if r["mode"] == m) for m in MODES}
    rate = {m: counts[m] / n if n else None for m in MODES}
    return {
        "n": n,
        "accuracy": rate["correct"],
        "omission_rate": rate["omission"],
        "hallucination_rate": rate["hallucination"],
        **{f"n_{m}": counts[m] for m in MODES},
    }


def caption_words(captions):
    lens = sorted(len(str(r["caption"]).split()) for r in captions.values())
    return round(sum(lens) / len(lens)) if lens else None


def compute(bench, results, judge):
    """Score every model with captions under `results`. Returns {"common": [...], "models"}."""
    results = Path(results)
    models = {}
    for cap_file in sorted((results / "captions").glob("*.jsonl")):
        name = cap_file.stem
        captions = latest_by(read_jsonl(cap_file), "id")
        good = current_verdicts(results / "runs" / f"{name}.jsonl", bench, captions, judge)
        if good:
            models[name] = {"verdicts": good, "words": caption_words(captions)}

    common = set(q["id"] for q in bench.questions)
    for m in models.values():
        common &= set(m["verdicts"])
    common = sorted(common)

    rows = []
    for name, m in models.items():
        rows.append(
            {
                "model": name,
                **tally([m["verdicts"][q] for q in common]),
                "own": tally(list(m["verdicts"].values())),
                "caption_words": m["words"],
            }
        )
    rows.sort(key=lambda r: (r["accuracy"] is None, -(r["accuracy"] or 0)))
    return {"judge": judge, "n_questions": len(bench.questions), "common": common, "rows": rows}


def render(board, fmt="table"):
    if fmt == "json":
        slim = {**board, "common": len(board["common"])}
        return json.dumps(slim, indent=1, ensure_ascii=False)
    n_common, n_all = len(board["common"]), board["n_questions"]
    head = (
        f"judge: {board['judge']} · scored on {n_common} of {n_all} questions "
        "(those every model has a current verdict for)\n\n"
    )
    lines = [
        "| # | model | accuracy | omission | hallucination | caption words | own coverage |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    pct = lambda x: "—" if x is None else f"{100 * x:.1f}"  # noqa: E731
    for i, r in enumerate(board["rows"], 1):
        lines.append(
            f"| {i} | {r['model']} | {pct(r['accuracy'])} | {pct(r['omission_rate'])} | "
            f"{pct(r['hallucination_rate'])} | {r['caption_words'] or '—'} | "
            f"{r['own']['n']}/{n_all} |"
        )
    return head + "\n".join(lines)
