"""Command line: download · caption · judge · board."""

import argparse
import json
import os
from pathlib import Path

from . import __version__

DEFAULT_DATA = os.environ.get("CAPTIONEXAM_DATA", "data")
DEFAULT_RESULTS = os.environ.get("CAPTIONEXAM_RESULTS", "results")
DEFAULT_JUDGE = "microsoft/phi-4"


def _bench(args):
    from .data import load_benchmark

    return load_benchmark(args.data)


def cmd_download(args):
    from .data import download

    path = download(args.data, repo_id=args.repo, revision=args.revision)
    print(f"Benchmark downloaded to {path}")


def cmd_caption(args):
    from .caption import run_captions
    from .prompts import CAPTION_PROMPT

    prompt = Path(args.prompt_file).read_text().strip() if args.prompt_file else CAPTION_PROMPT
    run_captions(
        _bench(args),
        Path(args.results) / "captions" / f"{args.name}.jsonl",
        model=args.model,
        name=args.name,
        prompt=prompt,
        base_url=args.base_url,
        api_key=args.api_key,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        workers=args.workers,
        images=args.images.split(",") if args.images else None,
        extra_body=json.loads(args.extra_body) if args.extra_body else None,
    )


def cmd_judge(args):
    from .judge import run_judge

    results = Path(args.results)
    names = args.name or sorted(p.stem for p in (results / "captions").glob("*.jsonl"))
    bench = _bench(args)
    for name in names:
        print(f"== {name}")
        run_judge(
            bench,
            results / "captions" / f"{name}.jsonl",
            results / "runs" / f"{name}.jsonl",
            judge_model=args.judge_model,
            judge_name=args.judge_name,
            base_url=args.judge_base_url,
            api_key=args.judge_api_key,
            workers=args.workers,
        )


def cmd_board(args):
    from .board import compute, render

    board = compute(_bench(args), args.results, args.judge_name or args.judge_model)
    print(render(board, args.format))


def main(argv=None):
    p = argparse.ArgumentParser(prog="captionexam", description=__doc__)
    p.add_argument("--version", action="version", version=f"captionexam {__version__}")
    p.add_argument("--data", default=DEFAULT_DATA, help="benchmark directory (default: data)")
    p.add_argument(
        "--results", default=DEFAULT_RESULTS, help="captions and verdicts (default: results)"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download", help="fetch the benchmark from the Hugging Face Hub")
    d.add_argument("--repo", default="Ki-Seki/CaptionExam")
    d.add_argument("--revision", default=None)
    d.set_defaults(func=cmd_download)

    c = sub.add_parser("caption", help="caption every image with a vision-language model")
    c.add_argument("--model", required=True, help="model id the server expects")
    c.add_argument("--name", required=True, help="name for this captioner in results/")
    c.add_argument("--base-url", default=None, help="OpenAI-compatible endpoint")
    c.add_argument("--api-key", default=None)
    c.add_argument("--prompt-file", default=None, help="replace the default caption prompt")
    c.add_argument("--max-tokens", type=int, default=4096)
    c.add_argument("--temperature", type=float, default=0.0)
    c.add_argument("--workers", type=int, default=8)
    c.add_argument("--images", default=None, help="comma-separated image ids (default: all)")
    c.add_argument(
        "--extra-body",
        default=None,
        help='JSON passed through to the server, e.g. \'{"chat_template_kwargs": '
        '{"enable_thinking": false}}\'',
    )
    c.set_defaults(func=cmd_caption)

    for name, help_ in (("judge", "judge captions"), ("board", "print the leaderboard")):
        s = sub.add_parser(name, help=help_)
        s.add_argument("--judge-model", default=DEFAULT_JUDGE)
        s.add_argument(
            "--judge-name", default=None, help="label stored with verdicts (default: judge model)"
        )
        if name == "judge":
            s.add_argument("--name", action="append", help="captioner(s) to judge (default: all)")
            s.add_argument("--judge-base-url", default=os.environ.get("CAPTIONEXAM_JUDGE_URL"))
            s.add_argument("--judge-api-key", default=None)
            s.add_argument("--workers", type=int, default=32)
            s.set_defaults(func=cmd_judge)
        else:
            s.add_argument("--format", choices=("table", "json"), default="table")
            s.set_defaults(func=cmd_board)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
