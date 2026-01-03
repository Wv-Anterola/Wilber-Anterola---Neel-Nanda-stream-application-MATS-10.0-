#!/usr/bin/env python3
"""
Layer sweep for activation patching.
"""

from pathlib import Path
import json

import numpy as np
import torch as t
from shared import load_model, manual_generate, tokenize_prompt

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DEVICE = "cuda" if t.cuda.is_available() else "cpu"
DTYPE = t.float16 if DEVICE == "cuda" else t.float32
ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "results" / "activation_patching"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEST_PROMPTS = [
    "What is 5 + 3?",
    "If Alice has 8 apples and gives 3 to Bob, how many does Alice have?",
    "What is 12 divided by 4?",
    "A rectangle has length 5 and width 3. What is its area?",
    "If a train travels 60 miles in 2 hours, what is its average speed?",
]

LAYERS_TO_TEST = [0, 1, 2, 3, 6, 10, 14, 18, 22, 26]

REASONING_TEMPLATE = (
    "Think step-by-step to solve this problem. Show your reasoning process.\n\n"
    "Question: {prompt}\n\n"
    "Let me think through this step by step:"
)
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


def count_generated_tokens(output_ids, prompt_len):
    return int(output_ids.shape[1] - prompt_len)


def get_cache_activations(model, tokens, layers):
    with t.no_grad():
        _, cache = model.run_with_cache(tokens)
    activations = {}
    for layer in layers:
        hook_name = f"blocks.{layer}.hook_resid_post"
        activations[str(layer)] = cache[hook_name].clone()
    return activations


def patch_activations_hook(layer_key, source_activations):
    def hook_fn(activations, hook):
        seq_len = min(activations.shape[1], source_activations[layer_key].shape[1])
        activations[:, :seq_len, :] = source_activations[layer_key][:, :seq_len, :]
        return activations

    return hook_fn


def generate_with_patching(model, input_ids, source_activations, patch_layers, max_tokens, eos_token_id, do_sample=False, temperature=1.0, top_p=1.0):
    hooks = []
    for layer in patch_layers:
        layer_key = str(layer)
        if layer_key in source_activations:
            hook_name = f"blocks.{layer}.hook_resid_post"
            hook_fn = patch_activations_hook(layer_key, source_activations)
            hooks.append((hook_name, hook_fn))

    return generate_tokens(
        model,
        input_ids,
        max_new_tokens=max_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=eos_token_id,
        fwd_hooks=hooks,
    )


def experiment_1_induce_reasoning(model, tokenizer, prompts):
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Induce Reasoning (Baseline to Reasoning)")
    print("=" * 70)
    print("\nExpected: Likely weak effect based on prior work.\n")

    results = []

    for prompt_idx, prompt in enumerate(prompts):
        print(f"\n[{prompt_idx + 1}/{len(prompts)}] {prompt[:50]}...")

        baseline_prompt = prompt
        reasoning_prompt = REASONING_TEMPLATE.format(prompt=prompt)

        baseline_ids = tokenize_prompt(tokenizer, baseline_prompt).to(model.cfg.device)
        reasoning_ids = tokenize_prompt(tokenizer, reasoning_prompt).to(model.cfg.device)

        source_acts = get_cache_activations(model, reasoning_ids, LAYERS_TO_TEST)

        with t.no_grad():
            baseline_output = generate_tokens(
                model,
                baseline_ids,
                max_new_tokens=200,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                eos_token_id=tokenizer.eos_token_id,
            )
        baseline_tokens_gen = count_generated_tokens(baseline_output, baseline_ids.shape[1])

        layer_results = {}
        for layer in LAYERS_TO_TEST:
            patched_output = generate_with_patching(
                model,
                baseline_ids,
                source_acts,
                [layer],
                max_tokens=200,
                eos_token_id=tokenizer.eos_token_id,
            )
            patched_tokens_gen = count_generated_tokens(patched_output, baseline_ids.shape[1])
            token_change = patched_tokens_gen - baseline_tokens_gen

            layer_key = str(layer)
            layer_results[layer_key] = {
                "tokens_generated": int(patched_tokens_gen),
                "token_change": int(token_change),
            }

            print(f"  Layer {layer:2d}: {patched_tokens_gen:3d} tokens ({token_change:+3d})")

        results.append({
            "prompt": prompt,
            "baseline_tokens": int(baseline_tokens_gen),
            "layer_results": layer_results,
        })

    output_file = OUTPUT_DIR / "exp1_induce_reasoning.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nExperiment 1 complete. Results: {output_file}")
    return results


def experiment_2_suppress_reasoning(model, tokenizer, prompts):
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Suppress Reasoning (Reasoning to Baseline)")
    print("=" * 70)
    print("\nExpected: Stronger suppression in early layers.\n")

    results = []

    for prompt_idx, prompt in enumerate(prompts):
        print(f"\n[{prompt_idx + 1}/{len(prompts)}] {prompt[:50]}...")

        baseline_prompt = prompt
        reasoning_prompt = REASONING_TEMPLATE.format(prompt=prompt)

        baseline_ids = tokenize_prompt(tokenizer, baseline_prompt).to(model.cfg.device)
        reasoning_ids = tokenize_prompt(tokenizer, reasoning_prompt).to(model.cfg.device)

        source_acts = get_cache_activations(model, baseline_ids, LAYERS_TO_TEST)

        with t.no_grad():
            reasoning_output = generate_tokens(
                model,
                reasoning_ids,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                eos_token_id=tokenizer.eos_token_id,
            )
        reasoning_tokens_gen = count_generated_tokens(reasoning_output, reasoning_ids.shape[1])

        layer_results = {}
        for layer in LAYERS_TO_TEST:
            patched_output = generate_with_patching(
                model,
                reasoning_ids,
                source_acts,
                [layer],
                max_tokens=200,
                eos_token_id=tokenizer.eos_token_id,
            )
            patched_tokens_gen = count_generated_tokens(patched_output, reasoning_ids.shape[1])
            token_change = patched_tokens_gen - reasoning_tokens_gen

            layer_key = str(layer)
            layer_results[layer_key] = {
                "tokens_generated": int(patched_tokens_gen),
                "token_change": int(token_change),
            }

            marker = '*' if token_change < -10 else ""
            print(f"  {marker} Layer {layer:2d}: {patched_tokens_gen:3d} tokens ({token_change:+3d})")

        results.append({
            "prompt": prompt,
            "reasoning_baseline_tokens": int(reasoning_tokens_gen),
            "layer_results": layer_results,
        })

    output_file = OUTPUT_DIR / "exp2_suppress_reasoning.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nExperiment 2 complete. Results: {output_file}")
    return results


def main():
    print("\n" + "=" * 70)
    print("Layer-level activation patching")
    print("=" * 70)

    model, tokenizer = load_model(MODEL_NAME, DEVICE, DTYPE, detail="layers")

    exp1_results = experiment_1_induce_reasoning(model, tokenizer, TEST_PROMPTS)
    exp2_results = experiment_2_suppress_reasoning(model, tokenizer, TEST_PROMPTS)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    avg_suppression_by_layer = {}
    for layer in LAYERS_TO_TEST:
        layer_key = str(layer)
        changes = [
            r["layer_results"][layer_key]["token_change"]
            for r in exp2_results
        ]
        avg_suppression_by_layer[layer] = float(np.mean(changes))

    best_layer = min(avg_suppression_by_layer.items(), key=lambda x: x[1])

    print(f"\nBest layer for suppression: Layer {best_layer[0]} ({best_layer[1]:.1f} tokens)")
    print("\nTop 5 layers:")
    for layer in sorted(avg_suppression_by_layer, key=avg_suppression_by_layer.get)[:5]:
        print(f"  Layer {layer:2d}: {avg_suppression_by_layer[layer]:+.1f} tokens")

    print("\n" + "=" * 70)
    print("Next: run head_analysis.py")
    print("=" * 70)


if __name__ == "__main__":
    t.set_grad_enabled(False)
    main()
