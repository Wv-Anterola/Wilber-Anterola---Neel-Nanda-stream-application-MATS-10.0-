#!/usr/bin/env python3
"""
Search patch configs and log the sweep.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
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

LOG_JSONL = OUTPUT_DIR / "quality_search_log.jsonl"
LOG_MD = OUTPUT_DIR / "quality_search_log.md"

REASONING_TEMPLATE = (
    "Think step-by-step to solve this problem. Show your reasoning process.\n\n"
    "Question: {prompt}\n\n"
    "Let me think through this step by step:"
)
TEST_PROBLEMS = MAIN_PROBLEMS

QUICK_PROBLEMS = TEST_PROBLEMS[:5]

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

HOOK_POINTS = {
    "resid_post": "blocks.{layer}.hook_resid_post",
    "resid_mid": "blocks.{layer}.hook_resid_mid",
    "attn_out": "blocks.{layer}.hook_attn_out",
}

LAYER_SETS = {
    "top3": [6, 2, 3],
    "early_0_6": [0, 1, 2, 3, 4, 5, 6],
    "single_6": [6],
}

ALPHAS = [1.0, 0.5]

DECODINGS = [
    {"name": "greedy", "do_sample": False, "temperature": 1.0, "top_p": 1.0},
    {"name": "sample", "do_sample": True, "temperature": 0.7, "top_p": 0.9},
]

MAX_NEW_TOKENS = 200


@dataclass
class Config:
    name: str
    hook_point: str
    layers: List[int]
    alpha: float
    decoding: Dict[str, float]


def build_configs() -> List[Config]:
    configs = []
    for hook_point in ["resid_post", "attn_out"]:
        for layer_name, layers in LAYER_SETS.items():
            for alpha in ALPHAS:
                for decoding in DECODINGS:
                    name = f"{hook_point}-{layer_name}-a{alpha}-{decoding['name']}"
                    configs.append(
                        Config(
                            name=name,
                            hook_point=hook_point,
                            layers=layers,
                            alpha=alpha,
                            decoding=decoding,
                        )
                    )
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
    numbers = re.findall(r"\b\d+\.?\d*\b", text)
    if numbers:
        return numbers[-1]
    return None


def count_reasoning_markers(text):
    text_lower = text.lower()
    return sum(1 for marker in REASONING_MARKERS if marker in text_lower)


def get_layer_outputs(model, tokens, layers, hook_point):
    with t.no_grad():
        _, cache = model.run_with_cache(tokens)
    outputs = {}
    for layer in layers:
        hook_name = HOOK_POINTS[hook_point].format(layer=layer)
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


def generate_with_intervention(
    model, prompt_ids, source_by_layer, layers, hook_point, alpha, eos_token_id, decoding
):
    hooks = []
    for layer in layers:
        hook_name = HOOK_POINTS[hook_point].format(layer=layer)
        hook_fn = patch_layer_hook(layer, source_by_layer, alpha)
        hooks.append((hook_name, hook_fn))

    return generate_tokens(
        model,
        prompt_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=decoding["do_sample"],
        temperature=decoding["temperature"],
        top_p=decoding["top_p"],
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


def log_result(stage, config, summary):
    entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stage": stage,
        "config": {
            "name": config.name,
            "hook_point": config.hook_point,
            "layers": config.layers,
            "alpha": config.alpha,
            "decoding": config.decoding,
        },
        "summary": summary,
    }
    with open(LOG_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    with open(LOG_MD, "a", encoding="utf-8") as f:
        f.write(
            f"- {entry['timestamp']} | {stage} | {config.name} | "
            f"verdict={summary['verdict']} | "
            f"token_red={summary['avg_token_reduction']:.1f} | "
            f"marker_red={summary['avg_marker_reduction']:.1f} | "
            f"maintained={summary['maintained_accuracy']:.2f}\n"
        )


def load_existing_log():
    if not LOG_JSONL.exists():
        return {}, {}, {}
    quick_by_name = {}
    full_by_name = {}
    config_by_name = {}
    with open(LOG_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            name = entry.get("config", {}).get("name")
            if not name:
                continue
            config_by_name[name] = entry.get("config", {})
            if entry.get("stage") == "quick":
                quick_by_name[name] = entry.get("summary")
            elif entry.get("stage") == "full":
                full_by_name[name] = entry.get("summary")
    return quick_by_name, full_by_name, config_by_name


def assess_config(model, tokenizer, problems, config):
    print("\n" + "=" * 70)
    print(f"TESTING CONFIG: {config.name}")
    print("=" * 70)
    print(
        f"hook_point={config.hook_point} layers={config.layers} "
        f"alpha={config.alpha} decoding={config.decoding['name']}"
    )

    results = []
    for i, problem in enumerate(problems, 1):
        print(f"\n[{i}/{len(problems)}] {problem['prompt']}")
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
            do_sample=config.decoding["do_sample"],
            temperature=config.decoding["temperature"],
            top_p=config.decoding["top_p"],
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
            config.decoding,
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
            "decoding": config.decoding,
        },
        "summary": summary,
        "detailed_results": results,
    }


def main():
    print("\n" + "=" * 70)
    print("Quality search v4")
    print("=" * 70)

    configs = build_configs()
    quick_log, full_log, _ = load_existing_log()
    model, tokenizer = load_model(MODEL_NAME, DEVICE, DTYPE)

    all_results = []
    for config in configs:
        if config.name in full_log:
            all_results.append(
                {
                    "config": {
                        "name": config.name,
                        "hook_point": config.hook_point,
                        "layers": config.layers,
                        "alpha": config.alpha,
                        "decoding": config.decoding,
                    },
                    "summary": full_log[config.name],
                }
            )
            continue

        try:
            if config.name in quick_log:
                quick_summary = quick_log[config.name]
                quick_result = {
                    "config": {
                        "name": config.name,
                        "hook_point": config.hook_point,
                        "layers": config.layers,
                        "alpha": config.alpha,
                        "decoding": config.decoding,
                    },
                    "summary": quick_summary,
                }
                print(f"SKIP quick (already logged): {config.name}")
            else:
                quick_result = assess_config(model, tokenizer, QUICK_PROBLEMS, config)
                log_result("quick", config, quick_result["summary"])
        except Exception as exc:
            log_result("quick_error", config, {"verdict": "ERROR", "avg_token_reduction": 0.0, "avg_marker_reduction": 0.0, "baseline_accuracy": 0.0, "intervened_accuracy": 0.0, "maintained_accuracy": 0.0})
            print(f"ERROR: {config.name} failed in quick pass: {exc}")
            continue

        if quick_result["summary"]["verdict"] in {"EFFECTIVE", "MODERATE"}:
            if config.name in full_log:
                all_results.append(
                    {
                        "config": {
                            "name": config.name,
                            "hook_point": config.hook_point,
                            "layers": config.layers,
                            "alpha": config.alpha,
                            "decoding": config.decoding,
                        },
                        "summary": full_log[config.name],
                    }
                )
                continue
            full_result = assess_config(model, tokenizer, TEST_PROBLEMS, config)
            log_result("full", config, full_result["summary"])
            all_results.append(full_result)
            if full_result["summary"]["verdict"] == "EFFECTIVE":
                output_path = OUTPUT_DIR / "quality_assessment_v4.json"
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(full_result, f, indent=2)
                print(f"\nstrong result FOUND: {config.name}")
                print(f"Saved: {output_path}")
                return
        else:
            all_results.append(quick_result)

    output_path = OUTPUT_DIR / "quality_assessment_v4.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"strategies": all_results}, f, indent=2)
    print(f"\nNo strong results found. Saved: {output_path}")


if __name__ == "__main__":
    t.set_grad_enabled(False)
    main()
