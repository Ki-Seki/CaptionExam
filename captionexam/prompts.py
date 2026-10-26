"""The default captioning prompt and the judge prompt."""

import hashlib
import json

from .data import NOT_MENTIONED

CAPTION_PROMPT = (
    "Describe this image in as much detail as you can. Cover every visible object, person, "
    "and piece of text, including exact wording, colors, materials, counts, positions, spatial "
    "relations, style, lighting, and background. Describe only what is visible."
)

JUDGE_PROMPT = """You answer multiple-choice questions about an image, but you cannot see the image:
you only have a DESCRIPTION of it, written by another model.

Answer strictly from what the description states or unambiguously entails (a named entity
implies its canonical attributes). If the description does not tell you, choose "{not_mentioned}".
Answer with option TEXT, not index. Answer quickly, no deliberation.

DESCRIPTION:
{caption}

QUESTION:
{question}
   options: {options}

Return JSON: {{"choice": [<option text>, ...]}}"""


def judge_prompt(caption, question, options):
    return JUDGE_PROMPT.format(
        not_mentioned=NOT_MENTIONED,
        caption=caption,
        question=question,
        options=json.dumps(options, ensure_ascii=False),
    )


def prompt_fp(prompt):
    return hashlib.sha1(prompt.encode()).hexdigest()[:12]
