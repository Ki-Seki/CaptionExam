#!/usr/bin/env python3
"""Caption and judge the benchmark with a series of open models on one GPU node.

Everything is pipelined. Weights download ahead in the background; each model starts as soon
as its own weights are on disk and enough GPUs are free, is served with vLLM on `tp` GPUs,
captions every image, and shuts down, with several models running side by side. One GPU runs
the judge, which grades each model as soon as its captions are done. A model that needs every
GPU on the node waits for the judge to finish its current model and hand its GPU back.

Everything is resumable: a model with complete captions is skipped, and judging only fills
gaps, so the same command can be rerun after an interruption.

models.json is a list of
  {"name": "qwen3-vl-8b", "repo": "Qwen/Qwen3-VL-8B-Instruct", "tp": 1,
   "extra_body": {...}, "vllm_args": [...]}            # last two optional
in the order to run them; put small models first and the largest last.

A model with a "base_url" is a hosted API instead, and "repo" is the model id it expects:
nothing is downloaded or served, no GPU is used, up to --api-parallel of them caption at once,
and the key comes from OPENAI_API_KEY. Any model may also set "image_detail", and
"max_tokens" or "temperature", where null leaves the field out of the request.

usage: python scripts/run_series.py --models models.json --weights /nvme/weights \
           [--data data] [--results results] [--on-update 'aws s3 sync results s3://...']
"""

import argparse
import json
import os
import queue
import shlex
import signal
import subprocess
import threading
import time
import urllib.request
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from captionexam.board import compute, render
from captionexam.caption import run_captions
from captionexam.data import latest_by, load_benchmark, read_jsonl
from captionexam.judge import run_judge
from captionexam.prompts import CAPTION_PROMPT, prompt_fp

READY_TIMEOUT = 3600  # loading the largest checkpoints from disk can take a while
JUDGE_PORT = 8099
POLL = 5


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(*parts):
    print(f"[{now()}]", *parts, flush=True)


def gpu_ids():
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        return [g for g in visible.split(",") if g]
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return []  # no GPUs here; hosted API models still run
    return out.stdout.split()


def gpu_name():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
        )
        return out.stdout.splitlines()[0].strip()
    except (OSError, IndexError):
        return None


def tool_version(cmd):
    try:
        return subprocess.run(cmd + ["--version"], capture_output=True, text=True).stdout.strip()
    except OSError:
        return None


def is_api(m):
    return bool(m.get("base_url"))


def gpus_for(m):
    return 0 if is_api(m) else m.get("tp", 1)


def done_future(value):
    f = Future()
    f.set_result(value)
    return f


def captions_done(path, bench, pfp):
    rows = latest_by(read_jsonl(path), "id")
    return all(
        rows.get(i, {}).get("prompt_fp") == pfp and rows[i].get("caption") for i in bench.images
    )


class Runner:
    def __init__(self, args):
        self.args = args
        self.bench = load_benchmark(args.data)
        self.results = Path(args.results)
        self.weights = Path(args.weights)
        self.prompt = (
            Path(args.prompt_file).read_text().strip() if args.prompt_file else CAPTION_PROMPT
        )
        self.pfp = prompt_fp(self.prompt)
        self.models = json.loads(Path(args.models).read_text())
        self.free = list(gpu_ids())
        self.total = len(self.free)
        self.lock = threading.Lock()
        self.update_lock = threading.Lock()
        self.summary = {}
        self.to_judge = queue.Queue()
        self.judge_pause = threading.Event()  # set: the judge must hand its GPU back
        self.captioning_done = threading.Event()
        self.fetches = ThreadPoolExecutor(max_workers=args.fetch_workers)
        self.vllm_version = tool_version(shlex.split(args.vllm))
        for sub in ("captions", "runs", "logs", "meta"):
            (self.results / sub).mkdir(parents=True, exist_ok=True)

    # ---- GPUs ----
    def take(self, n):
        with self.lock:
            if len(self.free) < n:
                return None
            gpus, self.free = self.free[:n], self.free[n:]
            return gpus

    def give(self, gpus):
        with self.lock:
            self.free.extend(gpus)

    def needs_whole_node(self, m):
        return self.total > 0 and gpus_for(m) == self.total

    # ---- weights ----
    def fetch(self, m):
        dest = self.weights / m["repo"]
        if (dest / ".captionexam_complete").exists():
            return dest
        t0 = time.time()
        if self.args.fetch_cmd:
            cmd = self.args.fetch_cmd.format(
                repo=m["repo"], dest=dest, revision=m.get("revision") or "main"
            )
            subprocess.run(cmd, shell=True, check=True)
        else:
            from huggingface_hub import HfApi, snapshot_download

            snapshot_download(
                m["repo"],
                revision=m.get("revision"),
                local_dir=dest,
                ignore_patterns=["*.pth", "original/*"],
            )
            sha = HfApi().model_info(m["repo"], revision=m.get("revision")).sha
            (dest / ".captionexam_revision").write_text(sha + "\n")
        (dest / ".captionexam_complete").write_text(now() + "\n")
        log(f"fetched {m['repo']} in {time.time() - t0:.0f}s")
        return dest

    # ---- serving ----
    def serve(self, repo, path, gpus, port, logfile, max_seqs, extra_args=()):
        # Capping sequences at the client's concurrency keeps hybrid (Mamba-style) models
        # within their state cache; vLLM's default of 1024 does not fit them on one GPU
        cmd = shlex.split(self.args.vllm) + [
            "serve",
            str(path),
            "--served-model-name",
            repo,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--max-model-len",
            str(self.args.max_model_len),
            "--gpu-memory-utilization",
            str(self.args.gpu_util),
            "--max-num-seqs",
            str(max_seqs),
            *extra_args,
        ]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(gpus)}
        out = open(logfile, "a")
        out.write(f"\n# {now()} {' '.join(cmd)}\n")
        out.flush()
        proc = subprocess.Popen(
            cmd, env=env, stdout=out, stderr=subprocess.STDOUT, start_new_session=True
        )
        deadline = time.time() + READY_TIMEOUT
        url = f"http://127.0.0.1:{port}/v1/models"
        while time.time() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"vLLM exited with {proc.returncode}; see {logfile}")
            try:
                with urllib.request.urlopen(url, timeout=5) as r:
                    if r.status == 200:
                        return proc
            except OSError:
                pass
            time.sleep(POLL)
        self.stop(proc)
        raise RuntimeError(f"vLLM not ready after {READY_TIMEOUT}s; see {logfile}")

    @staticmethod
    def stop(proc):
        if proc.poll() is not None:
            return
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()

    def on_update(self):
        if not self.args.on_update:
            return
        with self.update_lock:  # one sync at a time
            r = subprocess.run(self.args.on_update, shell=True)
            if r.returncode:
                log(f"on-update command failed with {r.returncode}")

    # ---- one captioner ----
    def caption_model(self, m, path, gpus):
        name, api = m["name"], is_api(m)
        started = now()
        max_tokens = m.get("max_tokens", self.args.max_tokens)
        temperature = m.get("temperature", 0.0)
        proc = None
        if api:
            base_url = m["base_url"]
        else:
            port = 8100 + int(gpus[0])
            logfile = self.results / "logs" / f"{name}.vllm.log"
            extra = ["--tensor-parallel-size", str(m.get("tp", 1)), *m.get("vllm_args", [])]
            proc = self.serve(
                m["repo"], path, gpus, port, logfile, self.args.caption_workers, extra
            )
            base_url = f"http://127.0.0.1:{port}/v1"
        try:
            run_captions(
                self.bench,
                self.results / "captions" / f"{name}.jsonl",
                model=m["repo"],
                name=name,
                prompt=self.prompt,
                base_url=base_url,
                max_tokens=max_tokens,
                temperature=temperature,
                workers=self.args.caption_workers,
                extra_body=m.get("extra_body"),
                image_detail=m.get("image_detail"),
                log=log,
            )
        finally:
            if proc is not None:
                self.stop(proc)
        rows = latest_by(read_jsonl(self.results / "captions" / f"{name}.jsonl"), "id")
        reasons = {}
        for r in rows.values():
            reasons[r.get("finish_reason")] = reasons.get(r.get("finish_reason"), 0) + 1
        rev = path / ".captionexam_revision" if path else None
        meta = {
            "name": name,
            "repo": m["repo"],
            "revision": rev.read_text().strip() if rev and rev.exists() else None,
            # what the server says it ran, e.g. the dated snapshot behind an API alias
            "served_models": sorted(
                {r["served_model"] for r in rows.values() if r.get("served_model")}
            ),
            "api": m.get("base_url"),
            "tp": None if api else m.get("tp", 1),
            "gpu": None if api else gpu_name(),
            "vllm": None if api else self.vllm_version,
            "vllm_args": m.get("vllm_args", []),
            "extra_body": m.get("extra_body"),
            "image_detail": m.get("image_detail"),
            "max_tokens": max_tokens,
            "max_model_len": None if api else self.args.max_model_len,
            "max_num_seqs": None if api else self.args.caption_workers,
            "caption_workers": self.args.caption_workers,
            "temperature": temperature,
            "prompt_fp": self.pfp,
            "captions": len(rows),
            "finish_reasons": reasons,
            "started": started,
            "finished": now(),
        }
        (self.results / "meta" / f"{name}.json").write_text(json.dumps(meta, indent=1) + "\n")

    def run_captioners(self):
        todo = []
        for m in self.models:
            if captions_done(
                self.results / "captions" / f"{m['name']}.jsonl", self.bench, self.pfp
            ):
                self.summary[m["name"]] = "already done"
                self.to_judge.put(m["name"])  # judging fills any gaps
            elif gpus_for(m) > self.total:
                self.summary[m["name"]] = f"needs {gpus_for(m)} GPUs, node has {self.total}"
            else:
                todo.append(m)
        log(f"{len(todo)} of {len(self.models)} captioners to run on {self.total} GPUs")
        weights = {
            m["name"]: done_future(None) if is_api(m) else self.fetches.submit(self.fetch, m)
            for m in todo
        }
        running, pending = [], list(todo)
        while pending or running:
            for item in list(running):
                t, m, gpus = item
                if not t.is_alive():
                    running.remove(item)
                    self.give(gpus)
            # The judge gives its GPU back only when a model that needs the whole node is next
            # in line with its weights on disk, or running
            head = pending[0] if pending else None
            whole = [m for _, m, _ in running if self.needs_whole_node(m)]
            if head and self.needs_whole_node(head) and weights[head["name"]].done():
                whole.append(head)
            if whole and not self.judge_pause.is_set():
                log(f"pausing the judge for {whole[0]['name']}")
                self.judge_pause.set()
            elif not whole and self.judge_pause.is_set():
                self.judge_pause.clear()
            for m in list(pending):
                fut = weights[m["name"]]
                if not fut.done():
                    continue
                try:
                    path = fut.result()
                except Exception as e:  # noqa: BLE001 - skip this model, keep the rest going
                    pending.remove(m)
                    log(f"! fetch failed for {m['repo']}: {e}")
                    self.summary[m["name"]] = f"fetch failed: {e}"
                    continue
                apis = sum(is_api(r) for _, r, _ in running)
                if is_api(m) and apis >= self.args.api_parallel:
                    continue
                gpus = self.take(gpus_for(m))
                if gpus is None:
                    continue
                pending.remove(m)
                t = threading.Thread(target=self.caption_worker, args=(m, path, gpus), daemon=True)
                t.start()
                running.append((t, m, gpus))
            time.sleep(POLL)

    def caption_worker(self, m, path, gpus):
        log(f"start {m['name']} " + (f"on GPUs {','.join(gpus)}" if gpus else "via its API"))
        try:
            self.caption_model(m, path, gpus)
            self.summary[m["name"]] = "captioned"
            self.to_judge.put(m["name"])
            log(f"done {m['name']}")
        except Exception as e:  # noqa: BLE001
            self.summary[m["name"]] = f"failed: {e}"
            log(f"! {m['name']} failed: {e}")
        self.on_update()

    # ---- the judge lane ----
    def judge_lane(self):
        judge = {"name": "judge", "repo": self.args.judge}
        weights = self.fetches.submit(self.fetch, judge)
        proc = gpus = None
        try:
            while True:
                idle = self.to_judge.empty()
                if proc is not None and (
                    self.judge_pause.is_set() or (idle and self.captioning_done.is_set())
                ):
                    self.stop(proc)
                    self.give(gpus)
                    proc = gpus = None
                    log("judge stopped, GPU released")
                if idle and self.captioning_done.is_set():
                    return
                if idle or self.judge_pause.is_set() or not weights.done():
                    time.sleep(POLL)
                    continue
                if proc is None:
                    gpus = self.take(1)
                    if gpus is None:
                        time.sleep(POLL)
                        continue
                    logfile = self.results / "logs" / "judge.vllm.log"
                    try:
                        proc = self.serve(
                            judge["repo"],
                            weights.result(),
                            gpus,
                            JUDGE_PORT,
                            logfile,
                            self.args.judge_workers,
                        )
                    except Exception as e:  # noqa: BLE001
                        self.give(gpus)
                        proc = gpus = None
                        log(f"! judge failed to start: {e}")
                        time.sleep(60)
                        continue
                    log(f"judge up on GPU {gpus[0]}")
                name = self.to_judge.get()
                log(f"judge {name}")
                try:
                    run_judge(
                        self.bench,
                        self.results / "captions" / f"{name}.jsonl",
                        self.results / "runs" / f"{name}.jsonl",
                        judge_model=judge["repo"],
                        base_url=f"http://127.0.0.1:{JUDGE_PORT}/v1",
                        workers=self.args.judge_workers,
                        log=log,
                    )
                except Exception as e:  # noqa: BLE001 - leave the gaps for a rerun
                    log(f"! judging {name} failed: {e}")
                self.write_board()
                self.on_update()
        finally:
            if proc is not None:
                self.stop(proc)
                self.give(gpus)

    def write_board(self):
        board = compute(self.bench, self.results, self.args.judge)
        (self.results / "board.md").write_text(render(board) + "\n")
        (self.results / "board.json").write_text(render(board, "json") + "\n")
        return board

    def main(self):
        lane = None
        if not self.args.skip_judge:
            lane = threading.Thread(target=self.judge_lane, daemon=True)
            lane.start()
        if self.args.judge_only:
            for m in self.models:
                if (self.results / "captions" / f"{m['name']}.jsonl").exists():
                    self.to_judge.put(m["name"])
        else:
            self.run_captioners()
        self.captioning_done.set()
        (self.results / "run_summary.json").write_text(json.dumps(self.summary, indent=1) + "\n")
        if lane is not None:
            lane.join()
            log("\n" + render(self.write_board()))
        self.fetches.shutdown(wait=True)
        self.on_update()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--models", required=True)
    p.add_argument("--weights", required=True, help="local directory for checkpoints")
    p.add_argument("--data", default="data")
    p.add_argument("--results", default="results")
    p.add_argument("--prompt-file", default=None)
    p.add_argument("--vllm", default="vllm", help="command that runs vLLM")
    p.add_argument("--judge", default="microsoft/phi-4")
    p.add_argument("--judge-workers", type=int, default=64)
    p.add_argument("--caption-workers", type=int, default=16)
    p.add_argument("--api-parallel", type=int, default=4, help="API models captioning at once")
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--max-model-len", type=int, default=16384)
    p.add_argument("--gpu-util", type=float, default=0.9)
    p.add_argument("--fetch-workers", type=int, default=3)
    p.add_argument(
        "--fetch-cmd",
        default=None,
        help="shell command that puts {repo} at {revision} into {dest} (default: Hugging Face)",
    )
    p.add_argument("--on-update", default=None, help="shell command run after each model")
    p.add_argument("--skip-judge", action="store_true")
    p.add_argument("--judge-only", action="store_true")
    return p.parse_args(argv)


if __name__ == "__main__":
    Runner(parse_args()).main()
