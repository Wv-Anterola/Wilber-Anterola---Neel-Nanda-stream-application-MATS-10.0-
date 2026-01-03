#!/usr/bin/env python3
"""
Compare intervention strategies and log results.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import json
import re

import torch as t
from problem_sets import SMALL_PROBLEMS
from shared import load_model, manual_generate, tokenize_prompt

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DEVICE = "cuda" if t.cuda.is_available() else "cpu"
DTYPE = t.float16 if DEVICE == "cuda" else t.float32
ROOT_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT_DIR / "results" / "activation_patching"
OUTPUT_DIR = ROOT_DIR / "results" / "activation_patching"
REASONING_TEMPLATE = (
    "Think step-by-step to solve this problem. Show your reasoning process.\n\n"
    "Question: {prompt}\n\n"
    "Let me think through this step by step:"
)
TEST_PROBLEMS = SMALL_PROBLEMS

REASONING_MARKERS = [
    "step",
    "first",
    "then",
    "next",
    "therefore",
    "so",
    "thus",
    "because",
    "since",
    "let",
    "consider",
    "we",
    "calculate",
]


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
def generate_tokens(model, input_ids, max_new_tokens, do_sample, temperature, top_p, eos_token_id, fwd_hooks=None):
    if fwd_hooks:
        return manual_generate(
            model,
            input_ids,
            max_new_tokens,
            do_sample,
            temperature,
            top_p,
            eos_token_id,
            fwd_hooks,
        )
    try:
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            return_type="tokens",
        )
        if isinstance(output, list):
            output = output[0]
        if isinstance(output, str):
            raise TypeError("generate returned string")
        return output
    except TypeError:
        return manual_generate(
            model,
            input_ids,
            max_new_tokens,
            do_sample,
            temperature,
            top_p,
            eos_token_id,
            [],
        )


def decode_response(tokenizer, output_ids, prompt_len):
    gen_ids = output_ids[0, prompt_len:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return text, int(gen_ids.numel())


def extract_answer(text):
    numbers = re.findall(r"\b\d+\.?\d*\b", text)
    if numbers:
        return numbers[-1]
    return None


def count_reasoning_markers(text):
    text_lower = text.lower()
    return sum(1 for marker in REASONING_MARKERS if marker in text_lower)


def get_head_outputs(model, tokens, layers):
    with t.no_grad():
        _, cache = model.run_with_cache(tokens)
    outputs = {}
    for layer in layers:
        hook_name = f"blocks.{layer}.attn.hook_z"
        outputs[layer] = cache[hook_name].clone()
    return outputs


def patch_head_hook(layer, head, source_attn):
    def hook_fn(attn_output, hook):
        seq_len = min(attn_output.shape[1], source_attn[layer].shape[1])
        attn_output[:, :seq_len, head, :] = source_attn[layer][:, :seq_len, head, :]
        return attn_output

    return hook_fn


def generate_with_intervention(model, prompt_ids, source_attn_by_layer, critical_heads, eos_token_id):
    hooks = []
    for layer, head in critical_heads:
        hook_name = f"blocks.{layer}.attn.hook_z"
        hook_fn = patch_head_hook(layer, head, source_attn_by_layer)
        hooks.append((hook_name, hook_fn))

    return generate_tokens(
        model,
        prompt_ids,
        max_new_tokens=200,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        eos_token_id=eos_token_id,
        fwd_hooks=hooks,
    )


def compute_layer_rankings(exp2_results):
    layer_avg = {}
    for result in exp2_results:
        for layer_str, vals in result["layer_results"].items():
            layer = int(layer_str)
            layer_avg.setdefault(layer, []).append(vals["token_change"])
    averaged = {layer: sum(changes) / len(changes) for layer, changes in layer_avg.items()}
    return sorted(averaged.items(), key=lambda x: x[1])


def compute_head_scores(head_level_results) -> Dict[Tuple[int, int], float]:
    head_scores = {}
    for layer_str, layer_results in head_level_results.items():
        layer = int(layer_str)
        if not layer_results:
            continue
        head_keys = layer_results[0]["head_results"].keys()
        for head_key in head_keys:
            changes = []
            for result in layer_results:
                head_entry = result["head_results"].get(head_key)
                if head_entry is not None:
                    changes.append(head_entry["token_change"])
            if changes:
                head_scores[(layer, int(head_key))] = float(sum(changes) / len(changes))
    return head_scores


def select_strategies(exp2_results, head_level_results):
    layer_rankings = compute_layer_rankings(exp2_results)
    head_scores = compute_head_scores(head_level_results)

    if not head_scores:
        raise RuntimeError("No head scores available to build strategies.")

    best_head = min(head_scores.items(), key=lambda x: x[1])[0]

    top_layers = [layer for layer, _ in layer_rankings[:3]]
    diverse_heads = []
    for layer in top_layers:
        candidates = [(head, score) for (lyr, head), score in head_scores.items() if lyr == layer]
        if not candidates:
            continue
        best_layer_head = min(candidates, key=lambda x: x[1])[0]
        diverse_heads.append((layer, best_layer_head))

    top_heads = sorted(head_scores.items(), key=lambda x: x[1])[:3]
    original_top3 = [pair for pair, _ in top_heads]

    return {
        "single_best": [best_head],
        "diverse_layers": diverse_heads,
        "original_top3": original_top3,
    }


def assess_strategy(model, tokenizer, problems, heads, label):
    print("\n" + "=" * 70)
    print(f"TESTING STRATEGY: {label}")
    print(f"Heads: {heads}")
    print("=" * 70)

    results = []
    layers = sorted(set(layer for layer, _ in heads))

    for i, problem in enumerate(problems, 1):
        print(f"\n[{i}/{len(problems)}] {problem['prompt']}")
        print(f"  Expected: {problem['answer']}")

        reasoning_prompt = REASONING_TEMPLATE.format(prompt=problem["prompt"])
        baseline_prompt = problem["prompt"]

        reasoning_ids = tokenize_prompt(tokenizer, reasoning_prompt).to(model.cfg.device)
        baseline_ids = tokenize_prompt(tokenizer, baseline_prompt).to(model.cfg.device)

        source_attn_by_layer = get_head_outputs(model, baseline_ids, layers)

        baseline_output = generate_tokens(
            model,
            reasoning_ids,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
        )
        baseline_text, baseline_tokens = decode_response(
            tokenizer,
            baseline_output,
            reasoning_ids.shape[1],
        )

        intervened_output = generate_with_intervention(
            model,
            reasoning_ids,
            source_attn_by_layer,
            heads,
            tokenizer.eos_token_id,
        )
        intervened_text, intervened_tokens = decode_response(
            tokenizer,
            intervened_output,
            reasoning_ids.shape[1],
        )

        baseline_answer = extract_answer(baseline_text)
        intervened_answer = extract_answer(intervened_text)

        baseline_markers = count_reasoning_markers(baseline_text)
        intervened_markers = count_reasoning_markers(intervened_text)

        baseline_correct = baseline_answer == problem["answer"]
        intervened_correct = intervened_answer == problem["answer"]

        print(
            f"  Base: {baseline_tokens}t, {baseline_markers}m, "
            f"ans={baseline_answer} {'pass' if baseline_correct else 'fail'}"
        )
        print(
            f"  Intv: {intervened_tokens}t ({intervened_tokens - baseline_tokens:+d}), "
            f"{intervened_markers}m ({intervened_markers - baseline_markers:+d}), "
            f"ans={intervened_answer} {'pass' if intervened_correct else 'fail'}"
        )

        results.append({
            "problem": problem["prompt"],
            "expected_answer": problem["answer"],
            "baseline": {
                "text": baseline_text,
                "tokens": baseline_tokens,
                "reasoning_markers": baseline_markers,
                "extracted_answer": baseline_answer,
                "correct": baseline_correct,
            },
            "intervened": {
                "text": intervened_text,
                "tokens": intervened_tokens,
                "reasoning_markers": intervened_markers,
                "extracted_answer": intervened_answer,
                "correct": intervened_correct,
            },
            "changes": {
                "token_reduction": baseline_tokens - intervened_tokens,
                "marker_reduction": baseline_markers - intervened_markers,
                "maintained_correctness": baseline_correct and intervened_correct,
            },
        })

    avg_token_reduction = sum(r["changes"]["token_reduction"] for r in results) / len(results)
    avg_marker_reduction = sum(r["changes"]["marker_reduction"] for r in results) / len(results)
    baseline_correct_count = sum(1 for r in results if r["baseline"]["correct"])
    intervened_correct_count = sum(1 for r in results if r["intervened"]["correct"])
    maintained_count = sum(1 for r in results if r["changes"]["maintained_correctness"])

    if avg_token_reduction > 20 and avg_marker_reduction > 2 and maintained_count >= len(results) * 0.7:
        verdict = "EFFECTIVE"
    elif avg_token_reduction > 10 and maintained_count >= len(results) * 0.5:
        verdict = "MODERATE"
    else:
        verdict = "WEAK"

    summary = {
        "avg_token_reduction": float(avg_token_reduction),
        "avg_marker_reduction": float(avg_marker_reduction),
        "baseline_accuracy": baseline_correct_count / len(results),
        "intervened_accuracy": intervened_correct_count / len(results),
        "maintained_accuracy": maintained_count / len(results),
        "verdict": verdict,
    }

    print("\nSUMMARY:", label)
    print("=" * 16)
    print(f"Token reduction: {summary['avg_token_reduction']:.1f}")
    print(f"Marker reduction: {summary['avg_marker_reduction']:.1f}")
    print(f"Baseline accuracy: {summary['baseline_accuracy']:.2f}")
    print(f"Intervened accuracy: {summary['intervened_accuracy']:.2f}")
    print(f"Maintained: {summary['maintained_accuracy']:.2f}")
    print(f"Verdict: {summary['verdict']}")

    return {
        "strategy": label,
        "heads": [{"layer": l, "head": h} for l, h in heads],
        "summary": summary,
        "detailed_results": results,
    }


def main():
    print("\n" + "=" * 70)
    print("Quality measurement v2")
    print("=" * 70)

    exp2_data = load_json(INPUT_DIR / "exp2_suppress_reasoning.json")
    head_level_results = load_json(INPUT_DIR / "head_level_results.json")

    strategies = select_strategies(exp2_data, head_level_results)

    model, tokenizer = load_model(MODEL_NAME, DEVICE, DTYPE)

    all_results = []
    for name, heads in strategies.items():
        if not heads:
            print(f"Warning: Strategy {name} has no heads. Skipping.")
            continue
        all_results.append(assess_strategy(model, tokenizer, TEST_PROBLEMS, heads, name))

    if not all_results:
        raise RuntimeError("No strategies produced results.")

    best = max(all_results, key=lambda r: (r["summary"]["verdict"], r["summary"]["avg_token_reduction"]))

    output_file = OUTPUT_DIR / "quality_assessment_v2.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "strategies": all_results,
            "best_strategy": {
                "name": best["strategy"],
                "summary": best["summary"],
                "heads": best["heads"],
            },
        }, f, indent=2)

    best_quality = {
        "verdict": best["summary"]["verdict"],
        "critical_heads": best["heads"],
        "summary": {
            "avg_token_reduction": best["summary"]["avg_token_reduction"],
            "avg_marker_reduction": best["summary"]["avg_marker_reduction"],
            "baseline_accuracy": best["summary"]["baseline_accuracy"],
            "intervened_accuracy": best["summary"]["intervened_accuracy"],
            "maintained_accuracy": best["summary"]["maintained_accuracy"],
        },
        "detailed_results": best["detailed_results"],
    }
    with open(OUTPUT_DIR / "quality_assessment.json", "w", encoding="utf-8") as f:
        json.dump(best_quality, f, indent=2)

    print("\n" + "=" * 70)
    print("STRATEGY COMPARISON")
    print("=" * 70)
    for result in all_results:
        summary = result["summary"]
        print(f"{result['strategy']}:")
        print(f"  Verdict: {summary['verdict']}")
        print(f"  Token reduction: {summary['avg_token_reduction']:.1f}")
        print(f"  Marker reduction: {summary['avg_marker_reduction']:.1f}")
        print(f"  Maintained accuracy: {summary['maintained_accuracy']:.2f}")

    print("\nBEST STRATEGY:", best["strategy"])
    print("Verdict:", best["summary"]["verdict"])


if __name__ == "__main__":
    t.set_grad_enabled(False)
    main()
