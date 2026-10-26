"""OpenAI-compatible chat calls, shared by the captioner and the judge.

Anything that speaks the OpenAI chat API works: vLLM, SGLang, LMDeploy, or a hosted API.
"""

import base64
import mimetypes
import os
import time

from openai import OpenAI

RETRIES = 3


def make_client(base_url=None, api_key=None, timeout=600):
    return OpenAI(
        base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
        api_key=api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY",
        timeout=timeout,
        max_retries=0,  # retried below, with backoff that also covers bad responses
    )


def image_url(path):
    mime = mimetypes.guess_type(str(path))[0] or "image/jpeg"
    data = base64.b64encode(open(path, "rb").read()).decode()
    return f"data:{mime};base64,{data}"


def chat(client, model, content, retries=RETRIES, **kwargs):
    """One chat completion with retries; returns the response."""
    for attempt in range(retries):
        try:
            return client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": content}], **kwargs
            )
        except Exception:  # noqa: BLE001 - network, rate limit, server overload
            if attempt == retries - 1:
                raise
            time.sleep(2 * 2**attempt)


def usage_of(response):
    u = getattr(response, "usage", None)
    return {
        "prompt_tokens": getattr(u, "prompt_tokens", None),
        "completion_tokens": getattr(u, "completion_tokens", None),
    }
