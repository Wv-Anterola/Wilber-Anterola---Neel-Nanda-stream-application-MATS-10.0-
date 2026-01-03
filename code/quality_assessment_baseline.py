#!/usr/bin/env python3
"""
Baseline intervention quality check.
"""

from pathlib import Path
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


def load_json(path):
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


def assess_intervention_quality(model, tokenizer, problems, critical_heads):
    print("\n" + "=" * 70)
    print("INTERVENTION QUALITY ASSESSMENT")
    print("=" * 70)
    print(f"\nTesting {len(critical_heads)} critical heads:")
    for layer, head in critical_heads:
        print(f"  - Layer {layer}, Head {head}")

    results = []
    layers = sorted(set(layer for layer, _ in critical_heads))

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
            critical_heads,
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
            f"  Baseline: {baseline_tokens} tokens, {baseline_markers} markers, "
            f"ans={baseline_answer} {'pass' if baseline_correct else 'fail'}"
        )
        print(
            f"  Intervened: {intervened_tokens} tokens ({intervened_tokens - baseline_tokens:+d}), "
            f"{intervened_markers} markers ({intervened_markers - baseline_markers:+d}), "
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

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    avg_token_reduction = sum(r["changes"]["token_reduction"] for r in results) / len(results)
    avg_marker_reduction = sum(r["changes"]["marker_reduction"] for r in results) / len(results)

    baseline_correct_count = sum(1 for r in results if r["baseline"]["correct"])
    intervened_correct_count = sum(1 for r in results if r["intervened"]["correct"])
    maintained_count = sum(1 for r in results if r["changes"]["maintained_correctness"])

    print(f"\nToken reduction: {avg_token_reduction:.1f} tokens")
    print(f"Marker reduction: {avg_marker_reduction:.1f} markers")

    print("\nCorrectness:")
    print(f"  Baseline: {baseline_correct_count}/{len(results)}")
    print(f"  Intervened: {intervened_correct_count}/{len(results)}")
    print(f"  Maintained: {maintained_count}/{len(results)}")

    if avg_token_reduction > 20 and avg_marker_reduction > 2 and maintained_count >= len(results) * 0.7:
        print("\nIntervention looks effective")
        verdict = "EFFECTIVE"
    elif avg_token_reduction > 10 and maintained_count >= len(results) * 0.5:
        print("\nIntervention looks moderate")
        verdict = "MODERATE"
    else:
        print("\nIntervention looks weak")
        verdict = "WEAK"

    output_file = OUTPUT_DIR / "quality_assessment.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "verdict": verdict,
            "critical_heads": [{"layer": l, "head": h} for l, h in critical_heads],
            "summary": {
                "avg_token_reduction": float(avg_token_reduction),
                "avg_marker_reduction": float(avg_marker_reduction),
                "baseline_accuracy": baseline_correct_count / len(results),
                "intervened_accuracy": intervened_correct_count / len(results),
                "maintained_accuracy": maintained_count / len(results),
            },
            "detailed_results": results,
        }, f, indent=2)

    print(f"\nSaved to {output_file}")
    return verdict, results


def main():
    print("\n" + "=" * 70)
    print("Quality measurement")
    print("=" * 70)

    critical_data = load_json(INPUT_DIR / "critical_heads.json")
    if not critical_data.get("top_heads"):
        raise RuntimeError("critical_heads.json is empty or invalid")

    critical_heads = [
        (h["layer"], h["head"]) for h in critical_data["top_heads"][:3]
    ]

    model, tokenizer = load_model(MODEL_NAME, DEVICE, DTYPE)
    assess_intervention_quality(model, tokenizer, TEST_PROBLEMS, critical_heads)

    print("\n" + "=" * 70)
    print("Next: run make_figures.py")
    print("=" * 70)


if __name__ == "__main__":
    t.set_grad_enabled(False)
    main()
