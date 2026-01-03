#!/usr/bin/env python3
"""
Compare prompt-end activations for v7 vs v8.
"""

from pathlib import Path
from typing import Dict, List
import json

import torch as t
import torch.nn.functional as F
from problem_sets import MAIN_PROBLEMS
from shared import load_model, tokenize_prompt

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DEVICE = "cuda" if t.cuda.is_available() else "cpu"
DTYPE = t.float16 if DEVICE == "cuda" else t.float32
ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "results" / "activation_patching"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_JSON = OUTPUT_DIR / "prompt_activation_comparison_v7_v8.json"

LAYERS = [6, 2, 3]

V7_TEMPLATE = (
    "Think step-by-step to solve this problem. Keep it short (3-5 steps).\n"
    "End with: Final answer: <number>.\n\n"
    "Question: {prompt}\n\n"
    "Let me think through this step by step:"
)

V8_TEMPLATE = (
    "Think step-by-step to solve this problem. Show your reasoning process.\n"
    "End with: Final answer: <number>.\n\n"
    "Question: {prompt}\n\n"
    "Let me think through this step by step:"
)
TEST_PROBLEMS = MAIN_PROBLEMS
def compare_prompt_pair(model, tokenizer, prompt: str) -> Dict:
    v7_prompt = V7_TEMPLATE.format(prompt=prompt)
    v8_prompt = V8_TEMPLATE.format(prompt=prompt)

    v7_ids = tokenize_prompt(tokenizer, v7_prompt).to(model.cfg.device)
    v8_ids = tokenize_prompt(tokenizer, v8_prompt).to(model.cfg.device)

    with t.no_grad():
        _, v7_cache = model.run_with_cache(v7_ids)
        _, v8_cache = model.run_with_cache(v8_ids)

    per_layer = {}
    for layer in LAYERS:
        key = f"blocks.{layer}.hook_resid_post"
        v7_act = v7_cache[key][:, -1, :]
        v8_act = v8_cache[key][:, -1, :]
        cos = F.cosine_similarity(v7_act, v8_act, dim=-1).item()
        l2 = t.norm(v7_act - v8_act, dim=-1).item()
        per_layer[str(layer)] = {"cosine": cos, "l2": l2}

    return {
        "prompt": prompt,
        "v7_prompt_len": int(v7_ids.shape[1]),
        "v8_prompt_len": int(v8_ids.shape[1]),
        "per_layer": per_layer,
    }


def summarize(results: List[Dict]) -> Dict:
    summary = {}
    for layer in LAYERS:
        cos_vals = [r["per_layer"][str(layer)]["cosine"] for r in results]
        l2_vals = [r["per_layer"][str(layer)]["l2"] for r in results]
        summary[str(layer)] = {
            "cosine_mean": float(sum(cos_vals) / len(cos_vals)),
            "cosine_min": float(min(cos_vals)),
            "cosine_max": float(max(cos_vals)),
            "l2_mean": float(sum(l2_vals) / len(l2_vals)),
            "l2_min": float(min(l2_vals)),
            "l2_max": float(max(l2_vals)),
        }
    summary["v7_prompt_len_mean"] = float(
        sum(r["v7_prompt_len"] for r in results) / len(results)
    )
    summary["v8_prompt_len_mean"] = float(
        sum(r["v8_prompt_len"] for r in results) / len(results)
    )
    return summary


def main():
    print("\nPROMPT ACTIVATION COMPARISON (v7 vs v8)")
    model, tokenizer = load_model(MODEL_NAME, DEVICE, DTYPE)

    results = []
    for i, item in enumerate(TEST_PROBLEMS, 1):
        print(f"[{i}/{len(TEST_PROBLEMS)}] {item['prompt']}")
        results.append(compare_prompt_pair(model, tokenizer, item["prompt"]))

    summary = summarize(results)
    out = {
        "model_name": MODEL_NAME,
        "device": DEVICE,
        "dtype": str(DTYPE),
        "layers": LAYERS,
        "summary": summary,
        "results": results,
    }

    OUTPUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Saved: {OUTPUT_JSON}")
    print("Summary:")
    for layer, stats in summary.items():
        if layer.isdigit():
            print(
                f"  Layer {layer}: cosine_mean={stats['cosine_mean']:.4f} "
                f"l2_mean={stats['l2_mean']:.2f}"
            )


if __name__ == "__main__":
    t.set_grad_enabled(False)
    main()
