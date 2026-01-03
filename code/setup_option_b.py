#!/usr/bin/env python3
"""
Setup and quick validation for baseline vs reasoning behavior.
"""

from pathlib import Path
import json
import sys

import torch as t
from shared import load_model, manual_generate, tokenize_prompt

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DEVICE = "cuda" if t.cuda.is_available() else "cpu"
DTYPE = t.float16 if DEVICE == "cuda" else t.float32
ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "results" / "activation_patching"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_MAX_TOKENS = 150
REASONING_MAX_TOKENS = 300
if DEVICE == "cpu":
    BASELINE_MAX_TOKENS = 50
    REASONING_MAX_TOKENS = 100

TEST_PROMPTS = [
    "What is 5 + 3?",
    "If Alice has 8 apples and gives 3 to Bob, how many does Alice have?",
    "What is 12 divided by 4?",
    "A rectangle has length 5 and width 3. What is its area?",
    "If a train travels 60 miles in 2 hours, what is its average speed?",
]

REASONING_TEMPLATE = (
    "Think step-by-step to solve this problem. Show your reasoning process.\n\n"
    "Question: {prompt}\n\n"
    "Let me think through this step by step:"
)

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
    "find",
]
def generate_tokens(model, input_ids, max_new_tokens, do_sample, temperature, top_p, eos_token_id):
    if DEVICE == "cpu" and do_sample:
        print("Warning: Sampling disabled on CPU to avoid NaNs; using greedy decoding.")
        do_sample = False
        temperature = 1.0
        top_p = 1.0
    try:
        output = model.generate(
            input_ids=input_ids,
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
        )


def decode_response(tokenizer, output_ids, prompt_len):
    gen_ids = output_ids[0, prompt_len:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return text, int(gen_ids.numel())


def has_reasoning_markers(text):
    text_lower = text.lower()
    count = sum(1 for marker in REASONING_MARKERS if marker in text_lower)
    return count >= 3


def validate_setup():
    print("\n" + "=" * 70)
    print("VALIDATION: Baseline vs Reasoning Responses")
    print("=" * 70 + "\n")

    model, tokenizer = load_model(MODEL_NAME, DEVICE, DTYPE, detail="layers")
    results = []

    if DEVICE == "cpu":
        print("Warning: CUDA not available. This will be slow and may fail.")
        print(
            f"Warning: Reducing max_new_tokens to {BASELINE_MAX_TOKENS}/{REASONING_MAX_TOKENS}."
        )

    for i, prompt in enumerate(TEST_PROMPTS, 1):
        print(f"\n[{i}/5] {prompt}")

        baseline_prompt = prompt
        reasoning_prompt = REASONING_TEMPLATE.format(prompt=prompt)

        baseline_ids = tokenize_prompt(tokenizer, baseline_prompt).to(model.cfg.device)
        reasoning_ids = tokenize_prompt(tokenizer, reasoning_prompt).to(model.cfg.device)

        baseline_out = generate_tokens(
            model,
            baseline_ids,
            max_new_tokens=BASELINE_MAX_TOKENS,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            eos_token_id=tokenizer.eos_token_id,
        )
        baseline_text, baseline_tokens = decode_response(
            tokenizer,
            baseline_out,
            baseline_ids.shape[1],
        )

        reasoning_out = generate_tokens(
            model,
            reasoning_ids,
            max_new_tokens=REASONING_MAX_TOKENS,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
        )
        reasoning_text, reasoning_tokens = decode_response(
            tokenizer,
            reasoning_out,
            reasoning_ids.shape[1],
        )

        ratio = reasoning_tokens / max(baseline_tokens, 1)
        has_reasoning = has_reasoning_markers(reasoning_text)

        print(f"\n  Baseline ({baseline_tokens} tokens):")
        print(f"  {baseline_text[:100]}...")
        print(f"\n  Reasoning ({reasoning_tokens} tokens):")
        print(f"  {reasoning_text[:100]}...")
        print(f"\n  Reasoning markers: {has_reasoning}")
        print(f"  Token ratio: {ratio:.2f}x")

        results.append({
            "prompt": prompt,
            "baseline": baseline_text,
            "reasoning": reasoning_text,
            "baseline_tokens": baseline_tokens,
            "reasoning_tokens": reasoning_tokens,
            "has_reasoning_markers": has_reasoning,
            "token_ratio": ratio,
        })

    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    avg_ratio = sum(r["token_ratio"] for r in results) / len(results)
    reasoning_count = sum(1 for r in results if r["has_reasoning_markers"])

    print(f"\nAverage token ratio: {avg_ratio:.2f}x")
    print(f"Responses with reasoning markers: {reasoning_count}/{len(results)}")

    if avg_ratio > 1.3 and reasoning_count >= 3:
        print("\nValidation passed")
        print("Proceed with the next runs.")
        decision = "PASS"
    else:
        print("\nValidation looks weak")
        print("PROCEED with caution")
        decision = "WEAK_PASS"

    output_file = OUTPUT_DIR / "validation_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "decision": decision,
            "results": results,
            "summary": {
                "avg_token_ratio": avg_ratio,
                "reasoning_marker_count": reasoning_count,
                "total_tests": len(results),
                "model_name": MODEL_NAME,
                "device": DEVICE,
                "dtype": str(DTYPE),
            },
        }, f, indent=2)

    print(f"\nResults saved to {output_file}")
    return decision


def main() -> int:
    print("\n" + "=" * 70)
    print("Setup and validation")
    print("=" * 70)

    try:
        validate_setup()
        print("\n" + "=" * 70)
        print("Next: run layer_level_patching.py")
        print("=" * 70)
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1
    finally:
        if t.cuda.is_available():
            t.cuda.empty_cache()


if __name__ == "__main__":
    t.set_grad_enabled(False)
    sys.exit(main())
