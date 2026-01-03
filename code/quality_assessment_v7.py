#!/usr/bin/env python3
"""
Quality check for the v7 prompt format.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import re
import time

import torch as t
from problem_sets import MAIN_PROBLEMS
from shared import load_model, manual_generate, tokenize_prompt

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DEVICE = "cuda" if t.cuda.is_available() else "cpu"
DTYPE = t.float16 if DEVICE == "cuda" else t.float32
ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "results" / "activation_patching"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_JSONL = OUTPUT_DIR / "quality_search_v7.jsonl"
LOG_MD = OUTPUT_DIR / "quality_search_v7.md"
OUTPUT_JSON = OUTPUT_DIR / "quality_assessment_v7.json"

REASONING_TEMPLATE = (
    "Think step-by-step to solve this problem. Keep it short (3-5 steps).\n"
    "End with: Final answer: <number>.\n\n"
    "Question: {prompt}\n\n"
    "Let me think through this step by step:"
)
TEST_PROBLEMS = MAIN_PROBLEMS

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
MAX_NEW_TOKENS = 280


@dataclass
class Config:
    name: str
    hook_point: str
    layers: List[int]
    alpha: float
    patch_mode: str
    max_patch_steps: Optional[int]


def build_configs() -> List[Config]:
    configs = [
        Config("prompt_all", "resid_post", [6, 2, 3], 1.0, "prompt_all", None),
        Config("prompt_all_layer6", "resid_post", [6], 1.0, "prompt_all", None),
        Config("prompt_all_layer3", "resid_post", [3], 1.0, "prompt_all", None),
    ]
    return configs
def generate_tokens(
    model,
    input_ids,
    max_new_tokens,
    do_sample,
    temperature,
    top_p,
    eos_token_id,
    fwd_hooks=None,
    max_patch_steps=None,
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
            max_patch_steps,
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
            None,
        )


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


def compute_patch_span(prompt_len, patch_mode) -> Tuple[int, int]:
    if patch_mode == "prompt_all":
        return 0, prompt_len
    if patch_mode.startswith("prompt_tail_"):
        tail = int(patch_mode.split("_")[-1])
        return max(0, prompt_len - tail), prompt_len
    if patch_mode.startswith("prompt_head_"):
        head = int(patch_mode.split("_")[-1])
        return 0, min(prompt_len, head)
    return 0, prompt_len


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


def generate_with_intervention(
    model,
    prompt_ids,
    source_by_layer,
    layers,
    hook_point,
    alpha,
    patch_span,
    eos_token_id,
    max_patch_steps,
):
    hooks = []
    for layer in layers:
        hook_name = HOOK_POINTS[hook_point].format(layer=layer)
        hook_fn = patch_layer_hook(layer, source_by_layer, alpha, patch_span[0], patch_span[1])
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
        max_patch_steps=max_patch_steps,
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


def log_result(config, summary):
    entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
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
    }
    with open(LOG_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    with open(LOG_MD, "a", encoding="utf-8") as f:
        f.write(
            f"- {entry['timestamp']} | {config.name} | "
            f"verdict={summary['verdict']} | "
            f"token_red={summary['avg_token_reduction']:.1f} | "
            f"marker_red={summary['avg_marker_reduction']:.1f} | "
            f"maintained={summary['maintained_accuracy']:.2f} | "
            f"marker_method={summary['marker_method']}\n"
        )


def assess_config(model, tokenizer, config):
    print("\n" + "=" * 70)
    print(f"TESTING CONFIG: {config.name}")
    print("=" * 70)
    print(
        f"hook_point={config.hook_point} layers={config.layers} alpha={config.alpha} "
        f"patch_mode={config.patch_mode} max_patch_steps={config.max_patch_steps}"
    )

    results = []
    for i, problem in enumerate(TEST_PROBLEMS, 1):
        print(f"\n[{i}/{len(TEST_PROBLEMS)}] {problem['prompt']}")
        print(f"  Expected: {problem['answer']}")

        reasoning_prompt = REASONING_TEMPLATE.format(prompt=problem["prompt"])
        baseline_prompt = problem["prompt"]

        reasoning_ids = tokenize_prompt(tokenizer, reasoning_prompt).to(model.cfg.device)
        baseline_ids = tokenize_prompt(tokenizer, baseline_prompt).to(model.cfg.device)

        source_by_layer = get_layer_outputs(model, baseline_ids, config.layers, config.hook_point)
        patch_span = compute_patch_span(reasoning_ids.shape[1], config.patch_mode)

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
            patch_span,
            tokenizer.eos_token_id,
            config.max_patch_steps,
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
    print("\nSUMMARY:", config.name)
    print("=" * 16)
    print(f"Token reduction: {summary['avg_token_reduction']:.1f}")
    print(f"Marker reduction: {summary['avg_marker_reduction']:.1f}")
    print(f"Baseline accuracy: {summary['baseline_accuracy']:.2f}")
    print(f"Intervened accuracy: {summary['intervened_accuracy']:.2f}")
    print(f"Maintained: {summary['maintained_accuracy']:.2f}")
    print(f"Verdict: {summary['verdict']}")

    return {
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


def main():
    print("\n" + "=" * 70)
    print("Quality assessment v7")
    print("=" * 70)

    configs = build_configs()
    model, tokenizer = load_model(MODEL_NAME, DEVICE, DTYPE)

    all_results = []
    for config in configs:
        result = assess_config(model, tokenizer, config)
        log_result(config, result["summary"])
        all_results.append(result)

        if result["summary"]["verdict"] == "EFFECTIVE":
            with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
            print(f"\nstrong result FOUND: {config.name}")
            print(f"Saved: {OUTPUT_JSON}")
            return

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"strategies": all_results}, f, indent=2)
    print(f"\nNo strong results found. Saved: {OUTPUT_JSON}")


if __name__ == "__main__":
    t.set_grad_enabled(False)
    main()
