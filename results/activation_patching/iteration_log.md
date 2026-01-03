# Iteration Log - Reasoning Style Transfer

## 2025-12-31
- v3 hook fix: switched quality patching to `attn.hook_result` to match head analysis; verdict remained WEAK (see `results/activation_patching/quality_assessment_v3.json`).
- v4 sweep: searched `resid_post` vs `attn_out`, alpha 1.0/0.5, layer sets (top3/early_0_6/single_6), greedy + sampling; best configs were MODERATE, none EFFECTIVE (see `results/activation_patching/quality_assessment_v4.json`, `results/activation_patching/quality_search_log.md`).

## 2026-01-01
- v5 targeted patching: prompt-span patching + max patch steps; best config `prompt_all` still WEAK due to low maintained accuracy and coarse marker count (see `results/activation_patching/quality_assessment_v5.json`, `results/activation_patching/quality_search_v5.md`).
- v6 prompt tuning: enforced "Step 1" + "Final answer"; outputs became brittle (e.g., "Step 2:" fragments) and accuracy dropped (see `results/activation_patching/quality_assessment_v6.json`).
- v7 strong attempt (in progress): shortened reasoning prompt, required explicit "Final answer", increased `MAX_NEW_TOKENS` to avoid truncation, improved answer normalization, and switched marker counting to enumerated steps + step-word + connectors. Script: `quality_assessment_v7.py`. Pending run + results.
## 2026-01-01 (continued)
- v7 result: prompt shortening + improved scoring increased baseline correctness but intervention expanded outputs; avg token reduction negative, verdict WEAK (see `results/activation_patching/quality_assessment_v7.json`).
- v8 result: reverted to long reasoning prompt, added explicit final answer, increased max tokens, improved scoring. Strong suppression recovered: avg token reduction 69.0, marker reduction 3.4, accuracy 10/10, verdict EFFECTIVE (see `results/activation_patching/quality_assessment_v8.json`).
## 2026-01-01 (expanded test set + stats)
- Expanded v8 test set to 25 problems; reran Qwen2.5-1.5B-Instruct. Results: avg token reduction 78.68, marker reduction 3.12, baseline 23/25, intervened 25/25, maintained 23/25; verdict EFFECTIVE (see `results/activation_patching/quality_assessment_v8.json`).
- Added paired t-tests via `compute_quality_stats.py`: token reduction t=10.22, p=3.22e-10; marker reduction t=8.51, p=1.04e-8 (see `results/activation_patching/quality_stats.json`).
- Attempted Qwen2.5-3B-Instruct with hf_xet; download succeeded but model load OOM on 8GB GPU.

## 2026-01-02 (prompt sensitivity + single-layer + head tests)
- Prompt activation comparison (v7 vs v8) at prompt end for layers 6/2/3: cosine similarity ~0.998, L2 distances small. Suggests v7â†’v8 improvement is not explained by large prompt-state differences at these layers (see `results/activation_patching/prompt_activation_comparison_v7_v8.json`).
- Single-layer v8 tests: layers 6, 2, and 3 individually reach EFFECTIVE suppression with similar accuracy, implying redundancy rather than strict complementarity (see `results/activation_patching/quality_assessment_v8_single_layers.json`).
- Head-level v8 tests (L3H4, L3H0-3) remain WEAK under v8 prompt, confirming head-only patches do not replicate layer-level effects (see `results/activation_patching/quality_assessment_v8_heads.json`).
- Attention category analysis (v8): prompt-end attention mass is dominated by general context; small but higher attention to instruction tokens ("Let me think...") and "Final answer" in layer 3 vs layers 2/6 (see `results/activation_patching/attention_category_analysis_v8.json`).
- Harder problem set (12 items): v8 achieves MODERATE suppression (token reduction 50.3, marker reduction 1.1) with 100% accuracy (see `results/activation_patching/quality_assessment_v8_hardset.json`).
- Logit lens probe at prompt end: correct answer token rank improves in layers 6 and 7 and again at the final layer (see `results/activation_patching/logit_lens_summary.json` and figures in `results/activation_patching/figures`).
- First generation token attention probe: forced "Final" token shows layers 2 and 6 emphasize instruction text and the "Final answer" phrase (see `results/activation_patching/generation_attention_summary.json` and figure in `results/activation_patching/figures`).
- Generation-phase probe (baseline vs intervened, 25 prompts): baseline outputs include "Final answer" in 18/25, intervened in 0/25, so pre-final and pre-answer analysis is baseline only. End-token comparisons show layers 2 and 6 shift attention toward generated tokens under intervention, while layer 3 shifts toward prompt context; end-token logit lens ranks improve sharply in late layers under intervention (see `results/activation_patching/generation_phase_probe.json` and new end-token figures).
