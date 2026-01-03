#!/usr/bin/env python3
"""
Prompt-end attention by token category.
"""

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
import json
import re

import torch as t
from problem_sets import MAIN_PROBLEMS
from shared import load_model, tokenize_prompt

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DEVICE = "cuda" if t.cuda.is_available() else "cpu"
DTYPE = t.float16 if DEVICE == "cuda" else t.float32
ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "results" / "activation_patching"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_JSON = OUTPUT_DIR / "attention_category_analysis_v8.json"

REASONING_TEMPLATE = (
    "Think step-by-step to solve this problem. Show your reasoning process.\n"
    "End with: Final answer: <number>.\n\n"
    "Question: {prompt}\n\n"
    "Let me think through this step by step:"
)
TEST_PROBLEMS = MAIN_PROBLEMS

LAYERS = [6, 2, 3]
CATEGORIES = ["final_answer", "question", "let_me_think", "numbers", "operators", "other"]


ANCHORS = {
    "final_answer": "Final answer:",
    "question": "Question:",
    "let_me_think": "Let me think through this step by step:",
}
def find_subsequence(haystack: List[int], needle: List[int]) -> List[int]:
    if not needle:
        return []
    n = len(needle)
    for i in range(len(haystack) - n + 1):
        if haystack[i : i + n] == needle:
            return list(range(i, i + n))
    return []


def token_categories(tokenizer, token_ids: List[int]) -> Dict[str, List[int]]:
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    categories = {k: set() for k in CATEGORIES}

    # anchor-based spans (fallbacks handle tokenization quirks)
    for key, anchor_text in ANCHORS.items():
        anchor_ids = tokenizer.encode(anchor_text, add_special_tokens=False)
        span = find_subsequence(token_ids, anchor_ids)
        for idx in span:
            categories[key].add(idx)

    # fallback for final_answer if anchor not found
    if not categories["final_answer"]:
        for i, tok in enumerate(tokens):
            if re.search(r"final|answer", tok, re.IGNORECASE):
                categories["final_answer"].add(i)

    # numbers and operators
    for i, tok in enumerate(tokens):
        if re.search(r"\d", tok):
            categories["numbers"].add(i)
        if re.search(r"[+\-*/=%]", tok) or "percent" in tok.lower():
            categories["operators"].add(i)

    # other
    all_marked = set().union(
        categories["final_answer"],
        categories["question"],
        categories["let_me_think"],
        categories["numbers"],
        categories["operators"],
    )
    categories["other"] = set(range(len(token_ids))) - all_marked

    return {k: sorted(v) for k, v in categories.items()}


def main():
    print("ATTENTION CATEGORY ANALYSIS (v8 prompt)")
    model, tokenizer = load_model(MODEL_NAME, DEVICE, DTYPE)

    n_heads = model.cfg.n_heads
    sums = {layer: t.zeros((n_heads, len(CATEGORIES)), device="cpu") for layer in LAYERS}
    counts = {layer: t.zeros((n_heads, len(CATEGORIES)), device="cpu") for layer in LAYERS}

    category_counts = defaultdict(list)
    prompt_lens = []

    for i, item in enumerate(TEST_PROBLEMS, 1):
        prompt = REASONING_TEMPLATE.format(prompt=item["prompt"])
        input_ids = tokenize_prompt(tokenizer, prompt).to(model.cfg.device)
        token_ids = input_ids[0].tolist()
        prompt_lens.append(len(token_ids))

        cats = token_categories(tokenizer, token_ids)
        for cat in CATEGORIES:
            category_counts[cat].append(len(cats[cat]))

        with t.no_grad():
            _, cache = model.run_with_cache(input_ids)

        q_pos = input_ids.shape[1] - 1
        for layer in LAYERS:
            patt = cache[f"blocks.{layer}.attn.hook_pattern"][0]
            # patt shape: [head, seq, seq]
            attn = patt[:, q_pos, :].float().cpu()
            for c_idx, cat in enumerate(CATEGORIES):
                idxs = cats[cat]
                if not idxs:
                    continue
                mass = attn[:, idxs].sum(dim=-1)
                sums[layer][:, c_idx] += mass
                counts[layer][:, c_idx] += 1.0

        print(f"[{i}/{len(TEST_PROBLEMS)}] {item['prompt']}")

    results = {
        "model_name": MODEL_NAME,
        "device": DEVICE,
        "dtype": str(DTYPE),
        "layers": LAYERS,
        "categories": CATEGORIES,
        "prompt_len_mean": float(sum(prompt_lens) / len(prompt_lens)),
        "category_token_count_mean": {k: float(sum(v) / len(v)) for k, v in category_counts.items()},
        "per_layer": {},
    }

    for layer in LAYERS:
        layer_data = {}
        mean = t.zeros_like(sums[layer])
        # avoid div by zero
        denom = counts[layer].clone()
        denom[denom == 0] = 1.0
        mean = sums[layer] / denom

        # category mean over heads
        cat_mean = mean.mean(dim=0)
        layer_data["category_mean"] = {CATEGORIES[i]: float(cat_mean[i]) for i in range(len(CATEGORIES))}

        # top heads per category
        top_heads = {}
        for c_idx, cat in enumerate(CATEGORIES):
            vals = mean[:, c_idx]
            ranked = t.topk(vals, k=min(3, vals.shape[0])).indices.tolist()
            top_heads[cat] = [
                {"head": int(h), "mean_mass": float(vals[h])} for h in ranked
            ]
        layer_data["top_heads"] = top_heads

        results["per_layer"][str(layer)] = layer_data

    OUTPUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved: {OUTPUT_JSON}")


if __name__ == "__main__":
    t.set_grad_enabled(False)
    main()
