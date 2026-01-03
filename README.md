# Reasoning Style Suppression via Activation Patching

## Overview

This repository contains the code, results, and figures from a study of reasoning‑style suppression in Qwen/Qwen2.5-1.5B-Instruct using activation patching. The central question is whether patching baseline activations into reasoning prompts reduces verbose step‑by‑step outputs while preserving answer correctness. Writeups and application materials are intentionally excluded.

## Key Findings (from included results)

- Layer sweeps show the strongest suppression effect in early layers 0–6, with weak effects in later layers.
- Under the long, explicit "Final answer" prompt (v8), patching layers [6, 2, 3] reduces output length by 78.68 tokens and reasoning markers by 3.12 on average; accuracy improves from 23/25 to 25/25 (paired tests p < 1e-8).
- Single‑layer patches at 6, 2, or 3 are individually effective, indicating redundancy rather than a strictly complementary multi‑layer circuit.
- Head‑level patches do not reproduce the layer‑level effect under the same prompt (negative result).
- A harder 12‑item set shows moderate suppression (token reduction 50.3, marker reduction 1.1) with 12/12 accuracy.

## Structure

- `code/`
  - Experiment scripts (layer sweeps, head tests, quality assessments, attention and logit‑lens probes)
- `results/activation_patching/`
  - JSON outputs, logs, and figures from the runs
  - Figures live in `results/activation_patching/figures/`

## Key Scripts

- `code/setup_validation.py` — model setup and baseline vs reasoning validation
- `code/layer_sweep.py` — layer sweep for causal suppression
- `code/head_analysis.py` — head‑level analysis
- `code/quality_assessment_v8.py` — main v8 intervention run
- `code/quality_assessment_single_layers.py` — single‑layer ablations
- `code/quality_assessment_hardset.py` — harder problem set test
- `code/prompt_activation_compare.py` — v7 vs v8 prompt‑end activation comparison
- `code/attention_category_analysis.py` — prompt‑end attention category probe
- `code/generation_phase_analysis.py` — generation‑phase probe (baseline vs intervened)
- `code/make_figures.py` — figure generation from results
- `code/shared.py` — shared model and tokenization helpers
- `code/problem_sets.py` — centralized problem lists

## Results

Primary outputs are in `results/activation_patching/`.

- `quality_assessment_v8.json` — main v8 run (25 problems)
- `quality_assessment_single_layers.json` — single‑layer ablations
- `quality_assessment_hardset.json` — hard set evaluation
- `prompt_activation_comparison_v7_v8.json` — v7 vs v8 prompt‑end comparison
- `generation_phase_probe.json` — generation‑phase probe outputs
- `figures/` — plots used in the writeup

## Environment

- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Library: `TransformerLens`
- Hardware: RTX 4070 8GB (float16)
- Decoding: greedy, max_new_tokens=320

## Reproduction Notes

This code assumes a local GPU and `transformer_lens` installed. Scripts write to `results/activation_patching/` relative to the repository root. Run scripts from the repository root to keep outputs in the expected location.

## License

No license specified. Treat as research code intended for reference.
