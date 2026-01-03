# Suppressing Reasoning Style via Activation Patching

## Overview

This repository contains code, results, and figures for a study of reasoning-style suppression in Qwen/Qwen2.5-1.5B-Instruct using activation patching. The core test patches direct-answer activations into step-by-step prompts to reduce verbose reasoning while preserving correctness. Writeups and application materials are intentionally excluded.

## Key Findings (from included results)

- Main effect: Patching layers 6, 2, and 3 reduces output length by 78.68 tokens and reasoning markers by 3.12 on average; accuracy improves from 23/25 to 25/25. Paired tests: t=10.22, p<1e-10, d=2.04.
- Layer sweep: Strongest suppression occurs in early layers 0-6, with weak effects by layer 14.
- Redundancy: Single-layer patches at 6, 2, or 3 are individually effective; combining layers yields only a marginal gain.
- v7 vs v8 prompts: Activation similarity at layers 6, 2, and 3 is 0.998. The v8 success reflects improved measurement (explicit "Final answer: X" format and longer token budget), not different mechanisms.
- Logit lens probe: At the final layer, answer rank improves from 74,970 to 132 and log-prob improves by 11.5, indicating a more linearly accessible answer under intervention.
- Head-level negative: Best single head (L3H4) reduces tokens by 3.9 versus 77.1 for the full layer, suggesting distributed computation.
- Hard set: A 12-item harder set shows token reduction 50.3 and marker reduction 1.1 with 12/12 accuracy.

## Structure

- `code/`
  - Experiment scripts (layer sweeps, head tests, quality assessments, attention and logit-lens probes)
- `results/activation_patching/`
  - JSON outputs, logs, and figures from the runs
  - Figures live in `results/activation_patching/figures/`

## Key Scripts

- `code/setup_validation.py` - model setup and baseline vs reasoning validation
- `code/layer_sweep.py` - layer sweep for causal suppression
- `code/head_analysis.py` - head-level analysis
- `code/quality_assessment_v8.py` - main v8 intervention run
- `code/quality_assessment_single_layers.py` - single-layer ablations
- `code/quality_assessment_hardset.py` - harder problem set test
- `code/prompt_activation_compare.py` - v7 vs v8 prompt-end activation comparison
- `code/attention_category_analysis.py` - prompt-end attention category probe
- `code/generation_phase_analysis.py` - generation-phase probe (baseline vs intervened)
- `code/make_figures.py` - figure generation from results
- `code/shared.py` - shared model and tokenization helpers
- `code/problem_sets.py` - centralized problem lists

## Results

Primary outputs are in `results/activation_patching/`.

- `quality_assessment.json` - selected run (v8)
- `quality_assessment_v8.json` - main v8 run (25 problems)
- `quality_assessment_v8_single_layers.json` - single-layer ablations
- `quality_assessment_v8_heads.json` - head-level test
- `quality_assessment_v8_hardset.json` - hard set evaluation
- `prompt_activation_comparison_v7_v8.json` - v7 vs v8 prompt-end comparison
- `logit_lens_summary.json` - logit lens probe summary
- `attention_category_analysis_v8.json` - prompt-end attention categories
- `generation_phase_probe.json` - generation-phase probe outputs
- `figures/` - plots used in the writeup

## Environment

- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Library: `TransformerLens`
- Hardware: RTX 4070 8GB (float16)
- Decoding: greedy, max_new_tokens=320

## Reproduction Notes

This code assumes a local GPU and `transformer_lens` installed. Scripts write to `results/activation_patching/` relative to the repository root. Run scripts from the repository root to keep outputs in the expected location.

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

## License

No license specified. Treat as research code intended for reference.
