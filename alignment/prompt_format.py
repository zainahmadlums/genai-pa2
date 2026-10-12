"""Prompt templates shared by SFT, DPO, and generation.

One template is chosen per experiment and used everywhere (training collators and
`scripts/generate.py`). The template name is stored next to every checkpoint in
`alignment_meta.json` so generation cannot silently use a different format.

Templates
---------
``alpaca`` (default for the Qwen/Qwen2.5-0.5B base model)::

    ### Instruction:
    {instruction}

    ### Response:
    {response}<eos>

``chatml`` (optional)::

    <|im_start|>user
    {instruction}<|im_end|>
    <|im_start|>assistant
    {response}<eos>

``chat_template`` is generation-only and defers to ``tokenizer.apply_chat_template``;
it is used for the SmolLM2-Instruct persona track, where TRL formats the training data
with the model's own chat template.
"""

from __future__ import annotations

TEMPLATES = {
    "alpaca": "### Instruction:\n{instruction}\n\n### Response:\n",
    "chatml": "<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n",
}

# Text that marks the start of a new turn. If a model keeps writing past its answer,
# generation output is cut at the first occurrence (same rule for every model).
STOP_STRINGS = {
    "alpaca": ["\n### Instruction:", "\n### Response:"],
    "chatml": ["<|im_end|>", "<|im_start|>"],
    "chat_template": [],
}

DEFAULT_TEMPLATE = "alpaca"
CHAT_TEMPLATE = "chat_template"


def format_prompt(instruction: str, template: str = DEFAULT_TEMPLATE) -> str:
    """Return the prompt text that precedes the assistant response."""

    if template not in TEMPLATES:
        raise ValueError(f"Unknown template {template!r}; expected one of {sorted(TEMPLATES)}")
    return TEMPLATES[template].format(instruction=instruction.strip())


def format_generation_prompt(tokenizer, instruction: str, template: str = DEFAULT_TEMPLATE) -> str:
    """Prompt text for generation, including the tokenizer chat template option."""

    if template == CHAT_TEMPLATE:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction.strip()}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return format_prompt(instruction, template)


def trim_at_stop_strings(text: str, template: str = DEFAULT_TEMPLATE) -> str:
    """Cut generated text at the first template turn marker and strip whitespace."""

    cut = len(text)
    for marker in STOP_STRINGS.get(template, []):
        position = text.find(marker)
        if position != -1:
            cut = min(cut, position)
    return text[:cut].strip()
