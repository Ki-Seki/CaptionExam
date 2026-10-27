"""Caption every benchmark image with a vision-language model."""

import concurrent.futures as cf
import threading
from datetime import datetime, timezone

from .client import chat, image_url, make_client, usage_of
from .data import append_jsonl, latest_by, read_jsonl
from .prompts import prompt_fp

FLUSH_EVERY = 8


def run_captions(
    bench,
    out_path,
    model,
    name,
    prompt,
    base_url=None,
    api_key=None,
    max_tokens=4096,
    temperature=0.0,
    workers=8,
    images=None,
    extra_body=None,
    log=print,
):
    """Write one caption per image to `out_path` (JSONL), skipping images already done.

    A caption made with a different prompt is redone, so changing the prompt never mixes
    two prompts in one file. Rows: {"id", "caption", "model", "prompt_fp", ...}.
    `extra_body` is passed through to the server, e.g. to switch off a model's thinking mode.
    """
    pfp = prompt_fp(prompt)
    done = {
        i for i, r in latest_by(read_jsonl(out_path), "id").items() if r.get("prompt_fp") == pfp
    }
    todo = [i for i in (images or bench.images) if i in bench.images and i not in done]
    log(f"{name}: {len(done)} captions already done, {len(todo)} to go")
    if not todo:
        return

    client = make_client(base_url, api_key)
    lock = threading.Lock()
    pending, failed = [], []

    def one(image_id):
        content = [
            {"type": "image_url", "image_url": {"url": image_url(bench.images[image_id])}},
            {"type": "text", "text": prompt},
        ]
        r = chat(
            client,
            model,
            content,
            max_tokens=max_tokens,
            temperature=temperature,
            extra_body=extra_body or None,
        )
        choice = r.choices[0]
        return {
            "id": image_id,
            "caption": (choice.message.content or "").strip(),
            "model": name,
            "served_model": model,
            "prompt_fp": pfp,
            "finish_reason": choice.finish_reason,
            "usage": usage_of(r),
            "at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(one, i): i for i in todo}
        for n, fut in enumerate(cf.as_completed(futures), 1):
            image_id = futures[fut]
            try:
                row = fut.result()
            except Exception as e:  # noqa: BLE001 - one failed image must not stop the run
                failed.append(image_id)
                log(f"  ! {image_id}: {e}")
                continue
            with lock:
                pending.append(row)
                if len(pending) >= FLUSH_EVERY:
                    append_jsonl(out_path, pending)
                    pending.clear()
            if n % 10 == 0 or n == len(todo):
                log(f"  {n}/{len(todo)} images")
    append_jsonl(out_path, pending)
    if failed:
        log(f"{len(failed)} images failed; rerun the same command to retry them")
