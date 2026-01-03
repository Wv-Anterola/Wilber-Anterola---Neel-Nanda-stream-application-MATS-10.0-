#!/usr/bin/env python3
"""
Harder problem set evaluation under the v8 prompt.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import json
import re

import torch as t
from problem_sets import HARD_PROBLEMS
from shared import load_model, tokenize_prompt
TEST_PROBLEMS = HARD_PROBLEMS

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DEVICE = "cuda" if t.cuda.is_available() else "cpu"
DTYPE = t.float16 if DEVICE == "cuda" else t.float32
ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "results" / "activation_patching"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_JSON = OUTPUT_DIR / "quality_assessment_v8_hardset.json"

REASONING_TEMPLATE = (
    "Think step-by-step to solve this problem. Show your reasoning process.\n"
    "End with: Final answer: <number>.\n\n"
    "Question: {prompt}\n\n"
    "Let me think through this step by step:"
)

HARD_PROBLEMS = [
    {"prompt": "If Alice has 12 apples, gives 5 to Bob, then buys 7 more, how many does she have now?", "answer": "14"},
    {"prompt": "A store sold 18 items in the morning and 27 in the afternoon from 100. How many are left?", "answer": "55"},
    {"prompt": "A tank had 80 liters, used 1/4 of it, then added 10 liters. How many liters now?", "answer": "70"},
    {"prompt": "If 3x + 4 = 28, what is x?", "answer": "8"},
    {"prompt": "If 5x - 10 = 35, what is x?", "answer": "9"},
    {"prompt": "A rectangle has length 12 and width 5. What is its perimeter?", "answer": "34"},
    {"prompt": "You buy 3 items at $7 each and use a $5 coupon. What is the total cost?", "answer": "16"},
    {"prompt": "A box has 6 rows of 7 chocolates. If you eat 10, how many are left?", "answer": "32"},
    {"prompt": "If you split 96 candies equally among 8 kids, then 2 kids leave. If you redistribute equally among the remaining kids, how many does each get?", "answer": "16"},
    {"prompt": "A train travels 120 miles in 3 hours and then 60 miles in 1 hour. What is its average speed for the whole trip?", "answer": "45"},
    {"prompt": "If 4x + 6 = 38, what is x?", "answer": "8"},
    {"prompt": "If 2x - 5 = 9, what is x?", "answer": "7"},
]

REASONING_CONNECTORS = [
    "first",
    "second",
    "third",
    "then",
    "next",
    "therefore",
    "thus",
    "because",
    "since",
]
MARKER_METHOD = "enumerated_steps+step_word+connectors_v1"

HOOK_POINTS = {
    "resid_post": "blocks.{layer}.hook_resid_post",
}

DECODING = {"name": "greedy", "do_sample": False, "temperature": 1.0, "top_p": 1.0}
MAX_NEW_TOKENS = 320


@dataclass
class Config:
    name: str
    hook_point: str
    layers: List[int]
    alpha: float
    patch_mode: str
    max_patch_steps: Optional[int]
def generate_tokens(model, input_ids, max_new_tokens, do_sample, temperature, top_p, eos_token_id, fwd_hooks=None):
    tokens = input_ids.clone()
    finished = t.zeros(tokens.shape[0], dtype=t.bool, device=tokens.device)
    for _ in range(max_new_tokens):
        logits = model.run_with_hooks(tokens, fwd_hooks=fwd_hooks) if fwd_hooks else model(tokens)
        next_logits = logits[:, -1, :] / max(temperature, 1e-6)
        next_token = t.argmax(next_logits, dim=-1, keepdim=True) if not do_sample else t.argmax(next_logits, dim=-1, keepdim=True)
        tokens = t.cat([tokens, next_token], dim=1)
        if eos_token_id is not None:
            finished |= next_token.squeeze(-1).eq(eos_token_id)
            if t.all(finished):
                break
    return tokens


def decode_response(tokenizer, output_ids, prompt_len):
    gen_ids = output_ids[0, prompt_len:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return text, int(gen_ids.numel())


def normalize_number(text):
    if text is None:
        return None
    value = text.strip().rstrip(".,;:")
    if re.fullmatch(r"-?\d+", value):
        return str(int(value))
    if re.fullmatch(r"-?\d+\.\d+", value):
        try:
            as_float = float(value)
        except ValueError:
            return value
        if abs(as_float - int(as_float)) < 1e-9:
            return str(int(as_float))
    return value


def extract_answer(text):
    patterns = [
        r"final answer[:\s]+(-?\d+\.?\d*)",
        r"answer is[:\s]+(-?\d+\.?\d*)",
        r"answer[:\s]+(-?\d+\.?\d*)",
    ]
    lowered = text.lower()
    for pattern in patterns:
        matches = re.findall(pattern, lowered)
        if matches:
            return normalize_number(matches[-1])
    numbers = re.findall(r"\b-?\d+\.?\d*\b", text)
    if numbers:
        return normalize_number(numbers[-1])
    return None


def count_reasoning_markers(text):
    text_lower = text.lower()
    count = 0
    count += len(re.findall(r"^\s*\d+[.)]", text, flags=re.MULTILINE))
    step_by_step = re.findall(r"step[- ]by[- ]step", text_lower)
    text_no_sbs = re.sub(r"step[- ]by[- ]step", " ", text_lower)
    count += len(step_by_step)
    count += len(re.findall(r"\bstep\b", text_no_sbs))
    connector_re = r"\b(" + "|".join(REASONING_CONNECTORS) + r")\b"
    count += len(re.findall(connector_re, text_lower))
    return count


def get_layer_outputs(model, tokens, layers, hook_point):
    with t.no_grad():
        _, cache = model.run_with_cache(tokens)
    outputs = {}
    for layer in layers:
        hook_name = HOOK_POINTS[hook_point].format(layer=layer)
        outputs[layer] = cache[hook_name].clone()
    return outputs


def patch_layer_hook(layer, source_by_layer, alpha, patch_start, patch_end):
    def hook_fn(activations, hook):
        seq_len = min(activations.shape[1], source_by_layer[layer].shape[1])
        end = min(seq_len, patch_end)
        start = min(patch_start, end)
        if start >= end:
            return activations
        if alpha >= 1.0:
            activations[:, start:end, :] = source_by_layer[layer][:, start:end, :]
        else:
            activations[:, start:end, :] = (
                (1.0 - alpha) * activations[:, start:end, :]
                + alpha * source_by_layer[layer][:, start:end, :]
            )
        return activations

    return hook_fn


def generate_with_intervention(model, prompt_ids, source_by_layer, layers, hook_point, alpha, eos_token_id):
    hooks = []
    for layer in layers:
        hook_name = HOOK_POINTS[hook_point].format(layer=layer)
        hook_fn = patch_layer_hook(layer, source_by_layer, alpha, 0, prompt_ids.shape[1])
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
        "marker_method": MARKER_METHOD,
    }


def main():
    print("\n" + "=" * 70)
    print("V8 HARD SET QUALITY TEST")
    print("=" * 70)

    model, tokenizer = load_model(MODEL_NAME, DEVICE, DTYPE)

    config = Config("v8_hardset", "resid_post", [6, 2, 3], 1.0, "prompt_all", None)

    results = []
    for i, problem in enumerate(HARD_PROBLEMS, 1):
        print(f"\n[{i}/{len(HARD_PROBLEMS)}] {problem['prompt']}")
        print(f"  Expected: {problem['answer']}")

        reasoning_prompt = REASONING_TEMPLATE.format(prompt=problem["prompt"])
        baseline_prompt = problem["prompt"]

        reasoning_ids = tokenize_prompt(tokenizer, reasoning_prompt).to(model.cfg.device)
        baseline_ids = tokenize_prompt(tokenizer, baseline_prompt).to(model.cfg.device)

        source_by_layer = get_layer_outputs(model, baseline_ids, config.layers, config.hook_point)

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
            config.layers,
            config.hook_point,
            config.alpha,
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
    out = {
        "config": {
            "name": config.name,
            "hook_point": config.hook_point,
            "layers": config.layers,
            "alpha": config.alpha,
            "patch_mode": config.patch_mode,
            "max_patch_steps": config.max_patch_steps,
            "decoding": DECODING,
        },
        "summary": summary,
        "detailed_results": results,
    }

    OUTPUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Saved: {OUTPUT_JSON}")
    print("Summary:")
    print(f"Token reduction: {summary['avg_token_reduction']:.1f}")
    print(f"Marker reduction: {summary['avg_marker_reduction']:.1f}")
    print(f"Baseline accuracy: {summary['baseline_accuracy']:.2f}")
    print(f"Intervened accuracy: {summary['intervened_accuracy']:.2f}")
    print(f"Maintained: {summary['maintained_accuracy']:.2f}")
    print(f"Verdict: {summary['verdict']}")


if __name__ == "__main__":
    t.set_grad_enabled(False)
    main()
