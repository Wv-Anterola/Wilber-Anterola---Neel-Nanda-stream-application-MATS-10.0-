#!/usr/bin/env python3
"""
Logit lens probe at the last prompt token.
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
    {"prompt": "What is 12 divided by 4?", "answer": "3"},
    {"prompt": "A rectangle has length 5 and width 3. What is its area?", "answer": "15"},
    {"prompt": "If 2x = 14, what is x?", "answer": "7"},
]


def format_reasoning_prompt(tokenizer, prompt):
    reasoning_prompt = (
        "Think step-by-step to solve this problem. Show your reasoning process.\n"
        "End with: Final answer: <number>.\n\n"
        f"Question: {prompt}\n\n"
        "Let me think through this step by step:"
    )
    messages = [{"role": "user", "content": reasoning_prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def main():
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE, dtype=DTYPE)
    model.eval()

    layer_count = model.cfg.n_layers
    layer_logp = {str(i): [] for i in range(layer_count)}
    layer_rank = {str(i): [] for i in range(layer_count)}

    results = []

    for ex in EXAMPLES:
        formatted = format_reasoning_prompt(tokenizer, ex["prompt"])
        tokens = tokenizer(formatted, return_tensors="pt").to(model.cfg.device)
        prompt_len = tokens.input_ids.shape[1]
        pos = prompt_len - 1

        with t.no_grad():
            _, cache = model.run_with_cache(tokens.input_ids)

        answer_ids = tokenizer.encode(ex["answer"], add_special_tokens=False)
        if not answer_ids:
            continue
        answer_id = answer_ids[0]

        per_layer = []
        for layer in range(layer_count):
            resid = cache[f"blocks.{layer}.hook_resid_post"][0, pos, :]
            ln_out = model.ln_final(resid)
            logits = model.unembed(ln_out)
            log_probs = t.log_softmax(logits, dim=-1)
            logp = float(log_probs[answer_id].item())
            rank = int((logits > logits[answer_id]).sum().item() + 1)

            layer_logp[str(layer)].append(logp)
            layer_rank[str(layer)].append(rank)
            per_layer.append({"layer": layer, "logp": logp, "rank": rank})

        results.append(
            {
                "prompt": ex["prompt"],
                "answer": ex["answer"],
                "answer_token_id": int(answer_id),
                "position": int(pos),
                "per_layer": per_layer,
            }
        )

    summary = []
    for layer in range(layer_count):
        logps = layer_logp[str(layer)]
        ranks = layer_rank[str(layer)]
        if logps:
            summary.append(
                {
                    "layer": layer,
                    "avg_logp": float(np.mean(logps)),
                    "avg_rank": float(np.mean(ranks)),
                }
            )

    out_path = OUTPUT_DIR / "logit_lens_summary.json"
    out_path.write_text(json.dumps({"summary": summary, "examples": results}, indent=2))
    print(f"Wrote {out_path}")

    try:
        import matplotlib.pyplot as plt

        layers = [d["layer"] for d in summary]
        avg_logp = [d["avg_logp"] for d in summary]
        avg_rank = [d["avg_rank"] for d in summary]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(layers, avg_logp, marker="o")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Avg log prob of correct answer token")
        ax.set_title("Logit Lens: Correct Answer Log Prob at Prompt End")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig_path = OUTPUT_DIR / "figures" / "logit_lens_logp.png"
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(layers, avg_rank, marker="o")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Avg rank of correct answer token")
        ax.set_title("Logit Lens: Correct Answer Rank at Prompt End")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig_path = OUTPUT_DIR / "figures" / "logit_lens_rank.png"
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print("Wrote logit lens figures")
    except Exception as exc:
        print(f"Plotting skipped: {exc}")


if __name__ == "__main__":
    main()
