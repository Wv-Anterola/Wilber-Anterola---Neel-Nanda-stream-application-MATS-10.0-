from __future__ import annotations

from typing import Optional

import torch as t
from transformer_lens import HookedTransformer


def load_model(model_name: str, device: str, dtype: t.dtype, *, detail: Optional[str] = None):
    print(f"Loading {model_name}...")
    model = HookedTransformer.from_pretrained(
        model_name,
        device=device,
        dtype=dtype,
    )
    model.eval()
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("Model tokenizer not available.")
    if detail == "layers":
        print(f"Loaded model ({model.cfg.n_layers} layers)")
    elif detail == "heads":
        print(f"Loaded model ({model.cfg.n_heads} heads per layer)")
    else:
        print("Loaded model")
    return model, tokenizer


def tokenize_prompt(tokenizer, prompt: str):
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        try:
            tokens = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        except TypeError:
            ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
            )
            tokens = t.tensor([ids], dtype=t.long)
    else:
        tokens = tokenizer(prompt, return_tensors="pt")["input_ids"]
    return tokens


def safe_softmax(logits: t.Tensor) -> t.Tensor:
    logits = t.nan_to_num(logits, nan=-1e9, posinf=1e9, neginf=-1e9)
    logits = t.clamp(logits, min=-1e4, max=1e4)
    logits = logits - logits.max(dim=-1, keepdim=True).values
    exp = t.exp(logits)
    denom = exp.sum(dim=-1, keepdim=True)
    return exp / t.clamp(denom, min=1e-9)


def sample_next_token(logits: t.Tensor, top_p: float) -> t.Tensor:
    probs = safe_softmax(logits)
    if top_p >= 1.0:
        return t.multinomial(probs, num_samples=1)
    sorted_probs, sorted_indices = t.sort(probs, descending=True)
    cumulative = t.cumsum(sorted_probs, dim=-1)
    sorted_probs[cumulative > top_p] = 0
    denom = sorted_probs.sum(dim=-1, keepdim=True)
    if (denom == 0).any():
        return t.argmax(probs, dim=-1, keepdim=True)
    sorted_probs = sorted_probs / denom
    sampled = t.multinomial(sorted_probs, num_samples=1)
    return sorted_indices.gather(-1, sampled)


def manual_generate(
    model,
    input_ids: t.Tensor,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    eos_token_id: Optional[int] = None,
    fwd_hooks=None,
    max_patch_steps: Optional[int] = None,
) -> t.Tensor:
    tokens = input_ids.clone()
    finished = t.zeros(tokens.shape[0], dtype=t.bool, device=tokens.device)
    for step in range(max_new_tokens):
        use_hooks = bool(fwd_hooks) and (max_patch_steps is None or step < max_patch_steps)
        if use_hooks:
            logits = model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)
        else:
            logits = model(tokens)
        next_logits = logits[:, -1, :] / max(temperature, 1e-6)
        if do_sample:
            next_token = sample_next_token(next_logits, top_p)
        else:
            next_token = t.argmax(next_logits, dim=-1, keepdim=True)
        tokens = t.cat([tokens, next_token], dim=1)
        if eos_token_id is not None:
            finished |= next_token.squeeze(-1).eq(eos_token_id)
            if t.all(finished):
                break
    return tokens


def generate_tokens(
    model,
    input_ids: t.Tensor,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    eos_token_id: Optional[int] = None,
    fwd_hooks=None,
    max_patch_steps: Optional[int] = None,
) -> t.Tensor:
    return manual_generate(
        model,
        input_ids,
        max_new_tokens,
        do_sample,
        temperature,
        top_p,
        eos_token_id,
        fwd_hooks,
        max_patch_steps,
    )
