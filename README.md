# CaptionExam

A hard, fine-grained benchmark for image captions. A **text-only judge** reads a model's caption, never the image, and answers multiple-choice questions about the image. Every question also offers *"not mentioned / indeterminable"*, so each verdict says whether the caption got a fact right, **left it out**, or **got it wrong**.

- **109 images, 3,099 questions** (15–42 per image, median 27): objects and counts, colors and materials, pose and orientation, layout and spatial relations, rendered text and its exact wording, typography, style, and background.
- **Diagnostic, not just a number**: accuracy splits into an omission rate and a hallucination rate, per question.
- **Cheap to run**: the default judge is [Phi-4](https://huggingface.co/microsoft/phi-4) served locally with vLLM; any OpenAI-compatible endpoint works for both the captioner and the judge.

Dataset: [huggingface.co/datasets/Ki-Seki/CaptionExam](https://huggingface.co/datasets/Ki-Seki/CaptionExam)

## How it works

```text
image ──► captioner under test ──► caption ──► text-only judge ──► verdict per question
                                                 ▲                  correct / omission / hallucination
                         question + options ─────┘
```

A question looks like this:

```json
{"id": "img_0001_q02", "image_id": "img_0001",
 "question": "What color is the body of the pickup truck parked under the carport?",
 "options": ["white or light silver", "dark blue or black", "red or maroon", "dark green",
             "not mentioned / indeterminable"],
 "answer": ["white or light silver"],
 "bbox": [453, 552, 622, 804]}
```

The judge sees the caption, the question, and the options, and must answer with the exact text of one option (enforced with JSON-schema decoding). Picking the answer is **correct**; picking *not mentioned* is an **omission**; picking any other option is a **hallucination**. Wrong options are chosen to be plausible for that image, so a judge cannot guess the answer without the caption.

## Quick start

```bash
pip install git+https://github.com/Ki-Seki/CaptionExam
captionexam download                      # benchmark -> ./data

# 1. Serve the judge (Phi-4, one GPU) on :8001
GPU=0 ./scripts/serve_judge.sh

# 2. Serve the model under test on :8000, e.g. with vLLM
vllm serve Qwen/Qwen3-VL-8B-Instruct --port 8000

# 3. Caption, judge, and print the leaderboard
captionexam caption --model Qwen/Qwen3-VL-8B-Instruct --name qwen3-vl-8b \
    --base-url http://localhost:8000/v1
captionexam judge --judge-base-url http://localhost:8001/v1
captionexam board
```

For a hosted model, point `--base-url` at its OpenAI-compatible endpoint and pass `--api-key` (or set `OPENAI_API_KEY`). Use `--prompt-file` to caption with your own prompt; the default asks for an exhaustive description of what is visible (see [`captionexam/prompts.py`](captionexam/prompts.py)).

## Results on disk

```text
results/
  captions/<name>.jsonl   one caption per image   {"id", "caption", "model", "prompt_fp", ...}
  runs/<name>.jsonl       append-only verdicts     {"question", "choice", "mode", "judge", ...}
```

Nothing is scored ahead of time: `captionexam board` folds the verdict logs on every run (the latest verdict per question wins). Each verdict stores fingerprints of the question and of the caption it was judged on, so:

- **rerunning** a command only fills gaps, such as failed calls or new images;
- **regenerating a caption** or **editing a question** makes the old verdicts stale, and the next `judge` run redoes just those;
- the main table ranks every model on the questions they **all** have current verdicts for; `own coverage` shows each model's total.

Captions may be plain text or structured JSON; the judge reads either as text. Commit `results/` alongside your report and anyone can recompute your scores.

## Data format

`questions.jsonl` holds one question per line:

| field | meaning |
|---|---|
| `id` | `<image_id>_q<NN>` |
| `image_id` | the image, `images/<image_id>.jpg` |
| `question` | a direct question about the image; it never mentions a caption |
| `options` | 3–6 options, exactly one of them `not mentioned / indeterminable` |
| `answer` | the correct option(s); never `not mentioned` |
| `bbox` | the region the question is about: `[y_min, x_min, y_max, x_max]`, normalized to 0–1000; `[0, 0, 1000, 1000]` for whole-image facts |

The questions never refer to a caption, so the same set can grade a generated image or an answer given while looking at the original.

## About the judge

Phi-4 runs with temperature 0 and a 256-token cap. We picked it over a proprietary judge for cost and reproducibility. In an earlier calibration over 11 captioners and about 3,100 questions, its answers agreed with a strong proprietary judge 75.5% of the time (κ = 0.62), and the two judges ranked the captioners almost identically (Spearman ρ = 0.89). Rankings are stable; individual verdicts are not ground truth. Scores are comparable only under the same judge, and the board filters verdicts by judge.

## License

Code: [Apache-2.0](LICENSE). Dataset: see the [dataset card](https://huggingface.co/datasets/Ki-Seki/CaptionExam).
