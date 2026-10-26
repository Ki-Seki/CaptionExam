import json
from types import SimpleNamespace

import pytest

from captionexam.data import NOT_MENTIONED, load_benchmark

QUESTIONS = [
    {
        "id": "img_0001_q01",
        "image_id": "img_0001",
        "question": "What color is the cat?",
        "options": ["black", "white", "orange", NOT_MENTIONED],
        "answer": ["orange"],
        "bbox": [100, 100, 900, 900],
    },
    {
        "id": "img_0001_q02",
        "image_id": "img_0001",
        "question": "What is written on the mug?",
        "options": ["HELLO", "HELL0", NOT_MENTIONED],
        "answer": ["HELLO"],
        "bbox": [0, 0, 1000, 1000],
    },
    {
        "id": "img_0002_q01",
        "image_id": "img_0002",
        "question": "How many chairs are there?",
        "options": ["two", "three", "four", NOT_MENTIONED],
        "answer": ["three"],
        "bbox": [0, 0, 1000, 1000],
    },
]


@pytest.fixture
def bench(tmp_path):
    root = tmp_path / "data"
    (root / "images").mkdir(parents=True)
    for i in ("img_0001", "img_0002"):
        (root / "images" / f"{i}.jpg").write_bytes(b"\xff\xd8fake-jpeg")
    (root / "questions.jsonl").write_text("".join(json.dumps(q) + "\n" for q in QUESTIONS))
    return load_benchmark(root)


class FakeClient:
    """Stands in for the OpenAI client; `reply(prompt)` decides the message content."""

    def __init__(self, reply):
        self.reply = reply
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, model, messages, **kwargs):
        content = messages[0]["content"]
        text = content if isinstance(content, str) else content[-1]["text"]
        self.calls.append({"model": model, "text": text, **kwargs})
        msg = SimpleNamespace(content=self.reply(text))
        return SimpleNamespace(
            choices=[SimpleNamespace(message=msg, finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=5),
        )
