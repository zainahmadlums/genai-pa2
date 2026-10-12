"""Judge model adapters. Each backend exposes ``name`` and ``call(prompt: str) -> str``.

``kaggle_benchmarks`` (default on Kaggle)
    Written against the public ``kaggle-benchmarks`` package, version 0.6.1
    (https://github.com/Kaggle/kaggle-benchmarks): ``kbench.llms`` is a dict of the models
    listed in ``LLMS_AVAILABLE``, ``kbench.kaggle.load_model(name)`` builds a model-proxy
    client for any other name, and ``llm.prompt(text)`` returns the reply text. Each call
    runs inside ``kbench.chats.new(..., orphan=True)`` so requests do not share history.

``openai_proxy`` (fallback)
    Talks to the same Kaggle model proxy directly through its OpenAI-compatible endpoint,
    ``$MODEL_PROXY_URL/openapi`` with ``$MODEL_PROXY_API_KEY`` (this is what
    ``kaggle_benchmarks.kaggle.model_proxy.ModelProxy`` does internally).

``mock``
    Offline pipeline check only. It never contacts a model and its verdicts are not results.

Neither real backend could be exercised outside Kaggle; see docs/KAGGLE_RUN_GUIDE.md.
"""

from __future__ import annotations

import json
import os
import re


class JudgeBackendError(RuntimeError):
    pass


class KaggleBenchmarksBackend:
    name = "kaggle_benchmarks"

    def __init__(self, model: str):
        try:
            import kaggle_benchmarks as kbench
        except Exception as exc:  # noqa: BLE001 - missing package or a failed import-time setup
            raise JudgeBackendError(
                f"kaggle_benchmarks could not be imported ({type(exc).__name__}: {exc}). Run this in a Kaggle "
                "benchmark notebook (https://www.kaggle.com/benchmarks/tasks/new) or use --backend openai_proxy."
            ) from exc
        self._kbench = kbench
        self.model = model
        available = dict(getattr(kbench, "llms", {}) or {})
        if model in available:
            self._llm = available[model]
        elif kbench.kaggle.is_configured():
            # Not pre-listed in LLMS_AVAILABLE: ask the model proxy for it by name.
            self._llm = kbench.kaggle.load_model(model)
        else:
            raise JudgeBackendError(
                "Kaggle model proxy is not configured (MODEL_PROXY_URL / MODEL_PROXY_API_KEY missing). "
                f"Models listed by kaggle_benchmarks: {sorted(available) or 'none'}"
            )

    @staticmethod
    def available_models() -> list[str]:
        import kaggle_benchmarks as kbench

        return sorted(getattr(kbench, "llms", {}) or {})

    def call(self, prompt: str) -> str:
        reply = None
        # In batch runs kaggle_benchmarks can swallow exceptions raised inside a chat
        # context, so a missing reply is turned back into an error here.
        with self._kbench.chats.new("pairwise_judge", orphan=True):
            reply = self._llm.prompt(prompt)
        if reply is None:
            raise JudgeBackendError("judge call produced no reply (request failed inside kaggle_benchmarks)")
        return reply if isinstance(reply, str) else str(reply)


class OpenAIProxyBackend:
    name = "openai_proxy"

    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None):
        try:
            import openai
        except ImportError as exc:  # pragma: no cover
            raise JudgeBackendError("The openai package is required for --backend openai_proxy") from exc
        base_url = base_url or os.environ.get("MODEL_PROXY_URL")
        api_key = api_key or os.environ.get("MODEL_PROXY_API_KEY")
        if not base_url or not api_key:
            raise JudgeBackendError("MODEL_PROXY_URL and MODEL_PROXY_API_KEY must be set for --backend openai_proxy")
        for suffix in ("/openapi", "/genai"):
            if base_url.endswith(suffix):
                base_url = base_url[: -len(suffix)]
                break
        self.model = model
        self._client = openai.OpenAI(api_key=api_key, base_url=base_url + "/openapi")

    def call(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model, messages=[{"role": "user", "content": prompt}]
        )
        if not response.choices or response.choices[0].message is None:
            raise JudgeBackendError("model proxy returned no choices")
        content = response.choices[0].message.content
        if not content:
            raise JudgeBackendError("model proxy returned empty content")
        return content


class MockBackend:
    """Deterministic stand-in: prefers the longer non-empty response, ties on equal length."""

    name = "mock"

    def __init__(self, model: str = "mock"):
        self.model = model
        self.calls = 0

    def call(self, prompt: str) -> str:
        self.calls += 1
        judgments = []
        items = re.findall(
            r"===== Item (\d+) =====\n(.*?)\n===== End of item \1 =====", prompt, flags=re.DOTALL
        )
        for number, body in items:
            response_a = body.split("[Response A]\n", 1)[1].split("\n[Response B]\n", 1)[0]
            response_b = body.split("\n[Response B]\n", 1)[1]
            if len(response_a) == len(response_b):
                winner = "tie"
            else:
                winner = "A" if len(response_a) > len(response_b) else "B"
            judgments.append({"item": int(number), "winner": winner, "reason": "mock backend: length heuristic"})
        return json.dumps({"judgments": judgments})


def make_backend(name: str, model: str):
    """Create a backend. ``auto`` prefers kaggle_benchmarks, then the OpenAI-compatible proxy."""

    if name == "mock":
        return MockBackend(model)
    if name == "kaggle_benchmarks":
        return KaggleBenchmarksBackend(model)
    if name == "openai_proxy":
        return OpenAIProxyBackend(model)
    if name == "auto":
        try:
            return KaggleBenchmarksBackend(model)
        except JudgeBackendError as first_error:
            try:
                return OpenAIProxyBackend(model)
            except JudgeBackendError as second_error:
                raise JudgeBackendError(
                    f"No judge backend available.\n- kaggle_benchmarks: {first_error}\n- openai_proxy: {second_error}"
                ) from second_error
    raise ValueError(f"Unknown backend {name!r}")
