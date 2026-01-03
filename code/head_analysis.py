#!/usr/bin/env python3
"""
Head-level patching analysis in selected layers.
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
INPUT_DIR = ROOT_DIR / "results" / "activation_patching"
OUTPUT_DIR = ROOT_DIR / "results" / "activation_patching"
TEST_PROMPTS = [
    "What is 5 + 3?",
    "If Alice has 8 apples and gives 3 to Bob, how many does Alice have?",
    "What is 12 divided by 4?",
]

REASONING_TEMPLATE = (
    "Think step-by-step to solve this problem. Show your reasoning process.\n\n"
    "Question: {prompt}\n\n"
    "Let me think through this step by step:"
)


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


def count_generated_tokens(output_ids, prompt_len):
    return int(output_ids.shape[1] - prompt_len)


def identify_critical_layers(exp2_results, top_n=3):
    all_layers = set()
    for result in exp2_results:
        all_layers.update(result["layer_results"].keys())

    layer_avg = {}
    for layer_str in all_layers:
        changes = [
            r["layer_results"][layer_str]["token_change"]
            for r in exp2_results
        ]
        layer_avg[int(layer_str)] = float(np.mean(changes))

    sorted_layers = sorted(layer_avg.items(), key=lambda x: x[1])
    critical = [layer for layer, _ in sorted_layers[:top_n]]

    print("\nCritical layers identified:")
    for i, (layer, avg_change) in enumerate(sorted_layers[:top_n], 1):
        print(f"  {i}. Layer {layer}: {avg_change:.1f} tokens")

    return critical


def get_head_outputs(model, tokens, layer):
    with t.no_grad():
        _, cache = model.run_with_cache(tokens)
    hook_name = f"blocks.{layer}.attn.hook_z"
    return cache[hook_name].clone()


def patch_single_head_hook(layer_key, head_idx, source_attn_output):
    def hook_fn(attn_output, hook):
        seq_len = min(attn_output.shape[1], source_attn_output[layer_key].shape[1])
        attn_output[:, :seq_len, head_idx, :] = source_attn_output[layer_key][:, :seq_len, head_idx, :]
        return attn_output

    return hook_fn


def generate_with_head_patching(model, input_ids, source_attn, layer, head, max_tokens, eos_token_id):
    hook_name = f"blocks.{layer}.attn.hook_z"
    hook_fn = patch_single_head_hook(str(layer), head, source_attn)

    return generate_tokens(
        model,
        input_ids,
        max_new_tokens=max_tokens,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        eos_token_id=eos_token_id,
        fwd_hooks=[(hook_name, hook_fn)],
    )


def analyze_critical_heads(model, tokenizer, critical_layers, prompts):
    print("\n" + "=" * 70)
    print("HEAD-LEVEL ANALYSIS")
    print("=" * 70)

    all_results = {}

    for layer in critical_layers:
        print(f"\n--- Analyzing Layer {layer} ({model.cfg.n_heads} heads) ---")
        layer_results = []

        for prompt_idx, prompt in enumerate(prompts):
            print(f"\n  Prompt {prompt_idx + 1}/{len(prompts)}: {prompt[:40]}...")

            reasoning_prompt = REASONING_TEMPLATE.format(prompt=prompt)
            baseline_prompt = prompt

            reasoning_ids = tokenize_prompt(tokenizer, reasoning_prompt).to(model.cfg.device)
            baseline_ids = tokenize_prompt(tokenizer, baseline_prompt).to(model.cfg.device)

            source_heads = {str(layer): get_head_outputs(model, baseline_ids, layer)}

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

            head_results = {}
            for head_idx in range(model.cfg.n_heads):
                patched_output = generate_with_head_patching(
                    model,
                    reasoning_ids,
                    source_heads,
                    layer,
                    head_idx,
                    max_tokens=200,
                    eos_token_id=tokenizer.eos_token_id,
                )

                patched_tokens_gen = count_generated_tokens(patched_output, reasoning_ids.shape[1])
                token_change = patched_tokens_gen - reasoning_tokens_gen

                head_key = str(head_idx)
                head_results[head_key] = {
                    "tokens_generated": int(patched_tokens_gen),
                    "token_change": int(token_change),
                }

            most_effective = min(head_results.items(), key=lambda x: x[1]["token_change"])
            print(f"    Most effective: Head {most_effective[0]} ({most_effective[1]['token_change']:+d} tokens)")

            layer_results.append({
                "prompt": prompt,
                "reasoning_baseline_tokens": int(reasoning_tokens_gen),
                "head_results": head_results,
            })

        all_results[str(layer)] = layer_results

    output_file = OUTPUT_DIR / "head_level_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {output_file}")
    return all_results


def identify_critical_heads(results, top_n=5):
    print("\n" + "=" * 70)
    print("CRITICAL HEADS IDENTIFICATION")
    print("=" * 70)

    head_effectiveness = {}

    for layer_str, layer_results in results.items():
        layer = int(layer_str)
        if not layer_results:
            continue
        for head_idx in range(len(layer_results[0]["head_results"])):
            head_key = str(head_idx)
            changes = []
            for result in layer_results:
                if head_key in result["head_results"]:
                    changes.append(result["head_results"][head_key]["token_change"])
            if changes:
                head_effectiveness[(layer, head_idx)] = float(np.mean(changes))

    sorted_heads = sorted(head_effectiveness.items(), key=lambda x: x[1])

    print(f"\nTop {top_n} most effective heads:")
    critical_heads = []

    for i, ((layer, head), avg_change) in enumerate(sorted_heads[:top_n], 1):
        print(f"  {i}. Layer {layer}, Head {head}: {avg_change:.1f} tokens")
        critical_heads.append((layer, head))

    output_file = OUTPUT_DIR / "critical_heads.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "top_heads": [
                {"layer": layer, "head": head, "avg_change": head_effectiveness[(layer, head)]}
                for layer, head in critical_heads
            ]
        }, f, indent=2)

    print(f"\nSaved to {output_file}")
    return critical_heads


def main():
    print("\n" + "=" * 70)
    print("Head-level analysis")
    print("=" * 70)

    exp2_data = load_json(INPUT_DIR / "exp2_suppress_reasoning.json")

    model, tokenizer = load_model(MODEL_NAME, DEVICE, DTYPE, detail="heads")
    critical_layers = identify_critical_layers(exp2_data, top_n=3)
    results = analyze_critical_heads(model, tokenizer, critical_layers, TEST_PROMPTS)
    identify_critical_heads(results, top_n=5)

    print("\n" + "=" * 70)
    print("Next: run quality_assessment_baseline.py")
    print("=" * 70)


if __name__ == "__main__":
    t.set_grad_enabled(False)
    main()
