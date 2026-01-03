#!/usr/bin/env python3
"""
Attention summary at the forced first-generation token.
"""

from pathlib import Path
import json

import numpy as np
import torch as t
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DEVICE = "cuda" if t.cuda.is_available() else "cpu"
DTYPE = t.float16 if DEVICE == "cuda" else t.float32
ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "results" / "activation_patching"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXAMPLES = [
    {"prompt": "What is 5 + 3?", "answer": "8"},
    {"prompt": "If Alice has 8 apples and gives 3 to Bob, how many does she have?", "answer": "5"},
]

LAYERS = [2, 3, 6]


def format_reasoning_prompt(tokenizer, prompt):
    reasoning_prompt = (
        "Think step-by-step to solve this problem. Show your reasoning process.\n"
        "End with: Final answer: <number>.\n\n"
        f"Question: {prompt}\n\n"
        "Let me think through this step by step:"
    )
    messages = [{"role": "user", "content": reasoning_prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def is_number_token(token_text):
    return any(ch.isdigit() for ch in token_text)


def is_operator_token(token_text):
    return any(ch in "+-*/=" for ch in token_text)


def main():
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE, dtype=DTYPE)
    model.eval()

    intro_text = (
        "Think step-by-step to solve this problem. Show your reasoning process.\n"
        "End with: Final answer: <number>.\n\n"
    )
    think_text = "Let me think through this step by step:"

    summary = {str(layer): [] for layer in LAYERS}
    details = []

    for ex in EXAMPLES:
        formatted = format_reasoning_prompt(tokenizer, ex["prompt"])
        prompt_tokens = tokenizer(formatted, return_tensors="pt").to(model.cfg.device)
        prompt_len = prompt_tokens.input_ids.shape[1]

        analysis_text = formatted + " Final"
        analysis_tokens = tokenizer(analysis_text, return_tensors="pt").to(model.cfg.device)
        target_pos = analysis_tokens.input_ids.shape[1] - 1

        # Build prompt segment lengths
        intro_tokens = tokenizer(intro_text, add_special_tokens=False)
        question_text = f"Question: {ex['prompt']}\n\n"
        question_tokens = tokenizer(question_text, add_special_tokens=False)
        think_tokens = tokenizer(think_text, add_special_tokens=False)

        intro_len = len(intro_tokens["input_ids"]) if isinstance(intro_tokens, dict) else len(intro_tokens)
        question_len = len(question_tokens["input_ids"]) if isinstance(question_tokens, dict) else len(question_tokens)
        think_len = len(think_tokens["input_ids"]) if isinstance(think_tokens, dict) else len(think_tokens)

        with t.no_grad():
            _, cache = model.run_with_cache(analysis_tokens.input_ids)

        layer_results = {}
        for layer in LAYERS:
            attn = cache[f"blocks.{layer}.attn.hook_pattern"][0, :, target_pos, :]
            attn_mean = attn.mean(dim=0)

            categories = {
                "instruction": 0.0,
                "final_answer_phrase": 0.0,
                "question": 0.0,
                "think": 0.0,
                "numbers": 0.0,
                "operators": 0.0,
                "prompt_other": 0.0,
            }

            for src_idx in range(attn_mean.shape[0]):
                weight = float(attn_mean[src_idx].item())
                token_text = tokenizer.decode([int(analysis_tokens.input_ids[0, src_idx])])

                if "Final" in token_text or "final" in token_text or "answer" in token_text:
                    categories["final_answer_phrase"] += weight
                elif is_number_token(token_text):
                    categories["numbers"] += weight
                elif is_operator_token(token_text):
                    categories["operators"] += weight
                elif src_idx < intro_len:
                    categories["instruction"] += weight
                elif src_idx < intro_len + question_len:
                    categories["question"] += weight
                elif src_idx < intro_len + question_len + think_len:
                    categories["think"] += weight
                else:
                    categories["prompt_other"] += weight

            layer_results[str(layer)] = categories
            summary[str(layer)].append(categories)

        details.append(
            {
                "prompt": ex["prompt"],
                "target_pos": int(target_pos),
                "layer_results": layer_results,
            }
        )

    # Aggregate
    aggregate = {}
    for layer in LAYERS:
        layer_key = str(layer)
        entries = summary[layer_key]
        if not entries:
            continue
        keys = entries[0].keys()
        aggregate[layer_key] = {k: float(np.mean([e[k] for e in entries])) for k in keys}

    out_path = OUTPUT_DIR / "generation_attention_summary.json"
    out_path.write_text(json.dumps({"aggregate": aggregate, "examples": details}, indent=2))
    print(f"Wrote {out_path}")

    try:
        import matplotlib.pyplot as plt

        categories = [
            "instruction",
            "final_answer_phrase",
            "question",
            "think",
            "numbers",
            "operators",
            "prompt_other",
        ]

        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(categories))
        width = 0.25

        for i, layer in enumerate(LAYERS):
            vals = [aggregate[str(layer)][c] for c in categories]
            ax.bar(x + i * width, vals, width, label=f"Layer {layer}")

        ax.set_xticks(x + width)
        ax.set_xticklabels(categories, rotation=30, ha="right")
        ax.set_ylabel("Avg attention mass")
        ax.set_title("First Generation Token Attention (Forced 'Final')")
        ax.legend()
        fig.tight_layout()
        fig_path = OUTPUT_DIR / "figures" / "generation_attention_categories.png"
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print("Wrote generation attention figure")
    except Exception as exc:
        print(f"Plotting skipped: {exc}")


if __name__ == "__main__":
    main()
