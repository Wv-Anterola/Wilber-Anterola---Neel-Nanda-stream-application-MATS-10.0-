#!/usr/bin/env python3
"""
Quality check with a tuned reasoning prompt.
"""

from pathlib import Path
import json
import re
import time

import torch as t
from problem_sets import SMALL_PROBLEMS
from shared import load_model, manual_generate, tokenize_prompt

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DEVICE = "cuda" if t.cuda.is_available() else "cpu"
DTYPE = t.float16 if DEVICE == "cuda" else t.float32
ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "results" / "activation_patching"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_JSONL = OUTPUT_DIR / "quality_search_v6.jsonl"
OUTPUT_JSON = OUTPUT_DIR / "quality_assessment_v6.json"

BASELINE_TEMPLATE = "Answer briefly and directly.\nQuestion: {prompt}\nAnswer:"
REASONING_TEMPLATE = (
    "Solve step-by-step. Use 'Step 1:', 'Step 2:', etc. End with 'Final answer: X'.\n"
    "Question: {prompt}\nStep 1:"
)
TEST_PROBLEMS = SMALL_PROBLEMS

REASONING_MARKERS = [
    "step",
    "final answer",
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

HOOK_POINT = "blocks.{layer}.hook_resid_post"
LAYERS = [6, 2, 3]
ALPHA = 1.0
MAX_NEW_TOKENS = 256
DECODING = {"do_sample": False, "temperature": 1.0, "top_p": 1.0}
def generate_tokens(
    model,
    input_ids,
    max_new_tokens,
    do_sample,
    temperature,
    top_p,
    eos_token_id,
    fwd_hooks=None,
):
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
    patterns = [
        r"final answer[:\s]+(-?\d+\.?\d*)",
        r"answer is[:\s]+(-?\d+\.?\d*)",
        r"answer[:\s]+(-?\d+\.?\d*)",
        r"equals\s+(-?\d+\.?\d*)",
        r"=\s*(-?\d+\.?\d*)",
    ]
    lowered = text.lower()
    for pat in patterns:
        matches = re.findall(pat, lowered)
        if matches:
            return normalize(matches[-1])
    nums = re.findall(r"-?\d+\.?\d*", text)
    if nums:
        return normalize(nums[-1])
    return None


def normalize(num_str):
    if num_str is None:
        return None
    s = str(num_str).strip()
    if s.endswith("."):
        s = s[:-1]
    try:
        val = float(s)
    except ValueError:
        return s
    if abs(val - int(val)) < 1e-6:
        return str(int(val))
    return str(val)


def count_reasoning_markers(text):
    text_lower = text.lower()
    return sum(1 for marker in REASONING_MARKERS if marker in text_lower)


def get_layer_outputs(model, tokens):
    with t.no_grad():
        _, cache = model.run_with_cache(tokens)
    outputs = {}
    for layer in LAYERS:
        hook_name = HOOK_POINT.format(layer=layer)
        outputs[layer] = cache[hook_name].clone()
    return outputs


def patch_layer_hook(layer, source_by_layer, alpha):
    def hook_fn(activations, hook):
        seq_len = min(activations.shape[1], source_by_layer[layer].shape[1])
        if alpha >= 1.0:
            activations[:, :seq_len, :] = source_by_layer[layer][:, :seq_len, :]
        else:
            activations[:, :seq_len, :] = (
                (1.0 - alpha) * activations[:, :seq_len, :]
                + alpha * source_by_layer[layer][:, :seq_len, :]
            )
        return activations

    return hook_fn


def generate_with_intervention(model, prompt_ids, source_by_layer, eos_token_id):
    hooks = []
    for layer in LAYERS:
        hook_name = HOOK_POINT.format(layer=layer)
        hook_fn = patch_layer_hook(layer, source_by_layer, ALPHA)
        hooks.append((hook_name, hook_fn))

    return generate_tokens(
        model,
        prompt_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=DECODING["do_sample"],
        temperature=DECODING["temperature"],
        top_p=DECODING["top_p"],
        eos_token_id=eos_token_id,
        fwd_hooks=hooks,
    )


def summarize_results(results):
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

    return {
        "avg_token_reduction": float(avg_token_reduction),
        "avg_marker_reduction": float(avg_marker_reduction),
        "baseline_accuracy": baseline_correct_count / len(results),
        "intervened_accuracy": intervened_correct_count / len(results),
        "maintained_accuracy": maintained_count / len(results),
        "verdict": verdict,
    }


def log_result(summary):
    entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "hook_point": HOOK_POINT,
            "layers": LAYERS,
            "alpha": ALPHA,
            "max_new_tokens": MAX_NEW_TOKENS,
        },
        "summary": summary,
    }
    with open(LOG_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def main():
    print("\n" + "=" * 70)
    print("Quality prompt tune v6")
    print("=" * 70)

    model, tokenizer = load_model(MODEL_NAME, DEVICE, DTYPE)

    results = []
    for i, problem in enumerate(TEST_PROBLEMS, 1):
        print(f"\n[{i}/{len(TEST_PROBLEMS)}] {problem['prompt']}")
        print(f"  Expected: {problem['answer']}")

        reasoning_prompt = REASONING_TEMPLATE.format(prompt=problem["prompt"])
        baseline_prompt = BASELINE_TEMPLATE.format(prompt=problem["prompt"])

        reasoning_ids = tokenize_prompt(tokenizer, reasoning_prompt).to(model.cfg.device)
        baseline_ids = tokenize_prompt(tokenizer, baseline_prompt).to(model.cfg.device)

        source_by_layer = get_layer_outputs(model, baseline_ids)

        baseline_output = generate_tokens(
            model,
            reasoning_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=DECODING["do_sample"],
            temperature=DECODING["temperature"],
            top_p=DECODING["top_p"],
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
            source_by_layer,
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

        results.append(
            {
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
            }
        )

    summary = summarize_results(results)
    log_result(summary)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)

    print("\nSUMMARY")
    print("=" * 16)
    print(f"Token reduction: {summary['avg_token_reduction']:.1f}")
    print(f"Marker reduction: {summary['avg_marker_reduction']:.1f}")
    print(f"Baseline accuracy: {summary['baseline_accuracy']:.2f}")
    print(f"Intervened accuracy: {summary['intervened_accuracy']:.2f}")
    print(f"Maintained: {summary['maintained_accuracy']:.2f}")
    print(f"Verdict: {summary['verdict']}")


if __name__ == "__main__":
    t.set_grad_enabled(False)
    main()
