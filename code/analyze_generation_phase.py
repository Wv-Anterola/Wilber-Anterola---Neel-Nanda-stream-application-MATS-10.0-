#!/usr/bin/env python3
"""
Generation-phase probe for baseline vs intervened outputs.
"""

from pathlib import Path
import json

import numpy as np
import torch as t
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer
from problem_sets import MAIN_PROBLEMS

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DEVICE = "cuda" if t.cuda.is_available() else "cpu"
DTYPE = t.float16 if DEVICE == "cuda" else t.float32
ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "results" / "activation_patching"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_NEW_TOKENS = 200
PATCH_LAYERS = [6, 2, 3]
ATTN_LAYERS = [2, 3, 6]
EXAMPLES = MAIN_PROBLEMS

PROMPT_CATEGORIES = ["final_answer", "question", "let_me_think", "numbers", "operators", "other"]
CATEGORIES = PROMPT_CATEGORIES + ["generated"]
ANCHORS = {
    "final_answer": "Final answer:",
    "question": "Question:",
    "let_me_think": "Let me think through this step by step:",
}


def format_reasoning_prompt(tokenizer, prompt):
    reasoning_prompt = (
        "Think step-by-step to solve this problem. Show your reasoning process.\n"
        "End with: Final answer: <number>.\n\n"
        f"Question: {prompt}\n\n"
        "Let me think through this step by step:"
    )
    messages = [{"role": "user", "content": reasoning_prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def format_baseline_prompt(tokenizer, prompt):
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def tokenize_ids(tokenizer, text):
    out = tokenizer(text, add_special_tokens=False)
    return out["input_ids"] if isinstance(out, dict) else out


def find_subsequence(haystack, needle):
    if not needle:
        return []
    n = len(needle)
    for i in range(len(haystack) - n + 1):
        if haystack[i : i + n] == needle:
            return list(range(i, i + n))
    return []


def is_number_token(token_text):
    return any(ch.isdigit() for ch in token_text)


def is_operator_token(token_text):
    return any(ch in "+-*/=" for ch in token_text)


def extract_digits(text):
    return "".join(ch for ch in text if ch.isdigit())


def token_categories(tokenizer, token_ids):
    categories = {k: set() for k in PROMPT_CATEGORIES}

    for key, anchor_text in ANCHORS.items():
        anchor_ids = tokenize_ids(tokenizer, anchor_text)
        span = find_subsequence(token_ids, anchor_ids)
        for idx in span:
            categories[key].add(idx)

    if not categories["final_answer"]:
        for i, tok in enumerate(token_ids):
            token_text = tokenizer.decode([tok])
            if "Final" in token_text or "final" in token_text or "answer" in token_text:
                categories["final_answer"].add(i)

    if not categories["question"]:
        for i, tok in enumerate(token_ids):
            token_text = tokenizer.decode([tok])
            if "Question" in token_text or "question" in token_text:
                categories["question"].add(i)

    if not categories["let_me_think"]:
        for i, tok in enumerate(token_ids):
            token_text = tokenizer.decode([tok]).lower()
            if "think" in token_text or "step" in token_text:
                categories["let_me_think"].add(i)

    assigned = set().union(*categories.values())

    for i, tok in enumerate(token_ids):
        if i in assigned:
            continue
        token_text = tokenizer.decode([tok])
        if is_number_token(token_text):
            categories["numbers"].add(i)
        elif is_operator_token(token_text) or "percent" in token_text.lower():
            categories["operators"].add(i)

    assigned = set().union(*categories.values())
    categories["other"] = set(range(len(token_ids))) - assigned

    category_by_idx = {}
    for cat, idxs in categories.items():
        for idx in idxs:
            category_by_idx[idx] = cat

    return category_by_idx


def locate_positions(token_texts, prompt_len, answer_str):
    final_pos = None
    for i in range(prompt_len, len(token_texts)):
        tok = token_texts[i]
        if "Final" in tok or "final" in tok:
            final_pos = i
            break

    answer_pos = None
    if final_pos is not None:
        for i in range(final_pos, len(token_texts)):
            digits = extract_digits(token_texts[i])
            if digits == answer_str:
                answer_pos = i
                break

    return final_pos, answer_pos


def last_non_eos(token_ids, eos_id):
    if not token_ids:
        return None
    idx = len(token_ids) - 1
    if eos_id is None:
        return idx
    while idx > 0 and token_ids[idx] == eos_id:
        idx -= 1
    return idx


def patch_hook(layer_idx, source_act):
    def hook_fn(activations, hook):
        seq_len = min(activations.shape[1], source_act.shape[1])
        activations[:, :seq_len, :] = source_act[:, :seq_len, :]
        return activations

    return hook_fn


def compute_attention_by_category(attn, pre_pos, prompt_len, category_by_idx, cat_to_idx):
    heads = attn.shape[0]
    per_head = np.zeros((heads, len(CATEGORIES)), dtype=np.float64)
    attn_pos = attn[:, pre_pos, :]

    for src_idx in range(attn_pos.shape[1]):
        if src_idx >= prompt_len:
            cat = "generated"
        else:
            cat = category_by_idx.get(src_idx, "other")
        idx = cat_to_idx[cat]
        per_head[:, idx] += attn_pos[:, src_idx].detach().cpu().numpy()

    return per_head


def summarize_attention(attn_sum, attn_count):
    summary = {}
    for layer, sum_vec in attn_sum.items():
        count = max(attn_count.get(layer, 0), 1)
        summary[layer] = {cat: float(sum_vec[i] / count) for i, cat in enumerate(CATEGORIES)}
    return summary


def summarize_logit(logit_list):
    summary = []
    for layer in sorted(logit_list.keys(), key=lambda x: int(x)):
        entries = logit_list[layer]
        if not entries:
            continue
        avg_logp = float(np.mean([e["logp"] for e in entries]))
        avg_rank = float(np.mean([e["rank"] for e in entries]))
        summary.append({"layer": int(layer), "avg_logp": avg_logp, "avg_rank": avg_rank})
    return summary


def plot_attention_bars(attn_summary, title, out_path):
    try:
        import matplotlib.pyplot as plt

        categories = CATEGORIES
        x = np.arange(len(categories))
        width = 0.25

        fig, ax = plt.subplots(figsize=(11, 5))
        for i, layer in enumerate(ATTN_LAYERS):
            vals = [attn_summary[str(layer)].get(cat, 0.0) for cat in categories]
            ax.bar(x + i * width, vals, width, label=f"Layer {layer}")
        ax.set_xticks(x + width)
        ax.set_xticklabels(categories, rotation=30, ha="right")
        ax.set_ylabel("Avg attention mass")
        ax.set_title(title)
        ax.legend()
        fig.tight_layout()
        fig_path = OUTPUT_DIR / "figures" / out_path
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
    except Exception as exc:
        print(f"Plotting skipped ({out_path}): {exc}")


def plot_logit(logit_summary, title, out_path, field):
    try:
        import matplotlib.pyplot as plt

        layers = [d["layer"] for d in logit_summary]
        vals = [d[field] for d in logit_summary]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(layers, vals, marker="o")
        ax.set_xlabel("Layer")
        ax.set_ylabel(f"Avg {field}")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig_path = OUTPUT_DIR / "figures" / out_path
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
    except Exception as exc:
        print(f"Plotting skipped ({out_path}): {exc}")


def plot_head_heatmap(head_matrix, title, out_path):
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(9, 6))
        sns.heatmap(
            head_matrix,
            ax=ax,
            cmap="viridis",
            cbar=True,
            xticklabels=CATEGORIES,
            yticklabels=[f"H{i}" for i in range(head_matrix.shape[0])],
        )
        ax.set_xlabel("Category")
        ax.set_ylabel("Head")
        ax.set_title(title)
        fig.tight_layout()
        fig_path = OUTPUT_DIR / "figures" / out_path
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
    except Exception as exc:
        print(f"Plotting skipped ({out_path}): {exc}")


def main():
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE, dtype=DTYPE)
    model.eval()

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    resid_names = [f"blocks.{layer}.hook_resid_post" for layer in range(n_layers)]
    attn_names = [f"blocks.{layer}.attn.hook_pattern" for layer in ATTN_LAYERS]
    cache_names = set(resid_names + attn_names)

    cat_to_idx = {cat: i for i, cat in enumerate(CATEGORIES)}

    attn_sum = {
        "baseline": {
            "pre_final": {str(layer): np.zeros(len(CATEGORIES)) for layer in ATTN_LAYERS},
            "pre_answer": {str(layer): np.zeros(len(CATEGORIES)) for layer in ATTN_LAYERS},
            "end": {str(layer): np.zeros(len(CATEGORIES)) for layer in ATTN_LAYERS},
        },
        "intervened": {
            "pre_final": {str(layer): np.zeros(len(CATEGORIES)) for layer in ATTN_LAYERS},
            "pre_answer": {str(layer): np.zeros(len(CATEGORIES)) for layer in ATTN_LAYERS},
            "end": {str(layer): np.zeros(len(CATEGORIES)) for layer in ATTN_LAYERS},
        },
    }
    attn_count = {
        "baseline": {
            "pre_final": {str(layer): 0 for layer in ATTN_LAYERS},
            "pre_answer": {str(layer): 0 for layer in ATTN_LAYERS},
            "end": {str(layer): 0 for layer in ATTN_LAYERS},
        },
        "intervened": {
            "pre_final": {str(layer): 0 for layer in ATTN_LAYERS},
            "pre_answer": {str(layer): 0 for layer in ATTN_LAYERS},
            "end": {str(layer): 0 for layer in ATTN_LAYERS},
        },
    }

    head_sum = {
        "baseline": {
            "pre_answer": {str(layer): np.zeros((n_heads, len(CATEGORIES))) for layer in ATTN_LAYERS},
            "end": {str(layer): np.zeros((n_heads, len(CATEGORIES))) for layer in ATTN_LAYERS},
        },
        "intervened": {
            "pre_answer": {str(layer): np.zeros((n_heads, len(CATEGORIES))) for layer in ATTN_LAYERS},
            "end": {str(layer): np.zeros((n_heads, len(CATEGORIES))) for layer in ATTN_LAYERS},
        },
    }
    head_count = {
        "baseline": {"pre_answer": {str(layer): 0 for layer in ATTN_LAYERS},
                     "end": {str(layer): 0 for layer in ATTN_LAYERS}},
        "intervened": {"pre_answer": {str(layer): 0 for layer in ATTN_LAYERS},
                       "end": {str(layer): 0 for layer in ATTN_LAYERS}},
    }

    logit_list = {
        "baseline": {"pre_answer": {str(layer): [] for layer in range(n_layers)},
                     "end": {str(layer): [] for layer in range(n_layers)}},
        "intervened": {"pre_answer": {str(layer): [] for layer in range(n_layers)},
                       "end": {str(layer): [] for layer in range(n_layers)}},
    }

    missing = {
        "baseline": {"missing_final": 0, "missing_answer": 0, "missing_end": 0},
        "intervened": {"missing_final": 0, "missing_answer": 0, "missing_end": 0},
    }

    examples_out = []

    for idx, ex in enumerate(EXAMPLES, 1):
        print(f"[{idx}/{len(EXAMPLES)}] {ex['prompt']}")
        baseline_prompt = format_baseline_prompt(tokenizer, ex["prompt"])
        reasoning_prompt = format_reasoning_prompt(tokenizer, ex["prompt"])

        baseline_tokens = tokenizer(baseline_prompt, return_tensors="pt").to(model.cfg.device)
        reasoning_tokens = tokenizer(reasoning_prompt, return_tensors="pt").to(model.cfg.device)
        prompt_len = reasoning_tokens.input_ids.shape[1]

        prompt_ids = reasoning_tokens.input_ids[0].tolist()
        category_by_idx = token_categories(tokenizer, prompt_ids)

        with t.no_grad():
            _, base_cache = model.run_with_cache(
                baseline_tokens.input_ids,
                names_filter=lambda name: name in {f"blocks.{layer}.hook_resid_post" for layer in PATCH_LAYERS},
            )

        source_acts = {
            layer: base_cache[f"blocks.{layer}.hook_resid_post"].detach().clone()
            for layer in PATCH_LAYERS
        }

        hooks = []
        for layer in PATCH_LAYERS:
            hook_name = f"blocks.{layer}.hook_resid_post"
            hooks.append((hook_name, patch_hook(layer, source_acts[layer])))

        outputs = {}
        with t.no_grad():
            outputs["baseline"] = model.generate(
                reasoning_tokens.input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )
            with model.hooks(fwd_hooks=hooks):
                outputs["intervened"] = model.generate(
                    reasoning_tokens.input_ids,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                )

        example_entry = {
            "prompt": ex["prompt"],
            "answer": ex["answer"],
            "prompt_len": int(prompt_len),
            "conditions": {},
        }

        for cond in ["baseline", "intervened"]:
            output_ids = outputs[cond][0]
            token_texts = [tokenizer.decode([int(tid)]) for tid in output_ids]
            final_pos, answer_pos = locate_positions(token_texts, prompt_len, ex["answer"])
            end_pos = last_non_eos(output_ids.tolist(), tokenizer.eos_token_id)
            pre_final_pos = final_pos - 1 if final_pos and final_pos > 0 else None
            pre_answer_pos = answer_pos - 1 if answer_pos and answer_pos > 0 else None

            if final_pos is None:
                missing[cond]["missing_final"] += 1
            if answer_pos is None:
                missing[cond]["missing_answer"] += 1
            if end_pos is None:
                missing[cond]["missing_end"] += 1

            with t.no_grad():
                if cond == "intervened":
                    with model.hooks(fwd_hooks=hooks):
                        _, cache = model.run_with_cache(
                            output_ids.unsqueeze(0),
                            names_filter=lambda name: name in cache_names,
                        )
                else:
                    _, cache = model.run_with_cache(
                        output_ids.unsqueeze(0),
                        names_filter=lambda name: name in cache_names,
                    )

            cond_entry = {
                "final_pos": int(final_pos) if final_pos is not None else None,
                "answer_pos": int(answer_pos) if answer_pos is not None else None,
                "pre_final_pos": int(pre_final_pos) if pre_final_pos is not None else None,
                "pre_answer_pos": int(pre_answer_pos) if pre_answer_pos is not None else None,
                "end_pos": int(end_pos) if end_pos is not None else None,
            }

            if pre_final_pos is not None:
                for layer in ATTN_LAYERS:
                    attn = cache[f"blocks.{layer}.attn.hook_pattern"][0]
                    per_head = compute_attention_by_category(
                        attn, pre_final_pos, prompt_len, category_by_idx, cat_to_idx
                    )
                    attn_sum[cond]["pre_final"][str(layer)] += per_head.mean(axis=0)
                    attn_count[cond]["pre_final"][str(layer)] += 1

            if pre_answer_pos is not None:
                for layer in ATTN_LAYERS:
                    attn = cache[f"blocks.{layer}.attn.hook_pattern"][0]
                    per_head = compute_attention_by_category(
                        attn, pre_answer_pos, prompt_len, category_by_idx, cat_to_idx
                    )
                    attn_sum[cond]["pre_answer"][str(layer)] += per_head.mean(axis=0)
                    attn_count[cond]["pre_answer"][str(layer)] += 1
                    head_sum[cond]["pre_answer"][str(layer)] += per_head
                    head_count[cond]["pre_answer"][str(layer)] += 1

                answer_ids = tokenizer.encode(ex["answer"], add_special_tokens=False)
                if answer_ids:
                    answer_id = answer_ids[0]
                    for layer in range(n_layers):
                        resid = cache[f"blocks.{layer}.hook_resid_post"][0, pre_answer_pos, :]
                        ln_out = model.ln_final(resid)
                        logits = model.unembed(ln_out)
                        log_probs = t.log_softmax(logits, dim=-1)
                        logp = float(log_probs[answer_id].item())
                        rank = int((logits > logits[answer_id]).sum().item() + 1)
                        logit_list[cond]["pre_answer"][str(layer)].append({"logp": logp, "rank": rank})

            if end_pos is not None:
                for layer in ATTN_LAYERS:
                    attn = cache[f"blocks.{layer}.attn.hook_pattern"][0]
                    per_head = compute_attention_by_category(
                        attn, end_pos, prompt_len, category_by_idx, cat_to_idx
                    )
                    attn_sum[cond]["end"][str(layer)] += per_head.mean(axis=0)
                    attn_count[cond]["end"][str(layer)] += 1
                    head_sum[cond]["end"][str(layer)] += per_head
                    head_count[cond]["end"][str(layer)] += 1

                answer_ids = tokenizer.encode(ex["answer"], add_special_tokens=False)
                if answer_ids:
                    answer_id = answer_ids[0]
                    for layer in range(n_layers):
                        resid = cache[f"blocks.{layer}.hook_resid_post"][0, end_pos, :]
                        ln_out = model.ln_final(resid)
                        logits = model.unembed(ln_out)
                        log_probs = t.log_softmax(logits, dim=-1)
                        logp = float(log_probs[answer_id].item())
                        rank = int((logits > logits[answer_id]).sum().item() + 1)
                        logit_list[cond]["end"][str(layer)].append({"logp": logp, "rank": rank})

            cond_entry["output_preview"] = tokenizer.decode(
                output_ids[-min(120, output_ids.shape[0]):],
                skip_special_tokens=True,
            )

            example_entry["conditions"][cond] = cond_entry

        examples_out.append(example_entry)

    summary = {"baseline": {}, "intervened": {}}
    for cond in ["baseline", "intervened"]:
        summary[cond]["counts"] = {
            "missing_final": missing[cond]["missing_final"],
            "missing_answer": missing[cond]["missing_answer"],
        }
        summary[cond]["attn_pre_final"] = summarize_attention(
            attn_sum[cond]["pre_final"], attn_count[cond]["pre_final"]
        )
        summary[cond]["attn_pre_answer"] = summarize_attention(
            attn_sum[cond]["pre_answer"], attn_count[cond]["pre_answer"]
        )
        summary[cond]["attn_end"] = summarize_attention(
            attn_sum[cond]["end"], attn_count[cond]["end"]
        )
        summary[cond]["logit_lens_pre_answer"] = summarize_logit(logit_list[cond]["pre_answer"])
        summary[cond]["logit_lens_end"] = summarize_logit(logit_list[cond]["end"])

        head_summary = {}
        for layer in ATTN_LAYERS:
            for phase in ["pre_answer", "end"]:
                count = max(head_count[cond][phase][str(layer)], 1)
                head_summary.setdefault(phase, {})[str(layer)] = {
                    "categories": CATEGORIES,
                    "mean_mass": (head_sum[cond][phase][str(layer)] / count).tolist(),
                }
        summary[cond]["head_by_phase"] = head_summary

    out = {
        "model_name": MODEL_NAME,
        "device": DEVICE,
        "dtype": str(DTYPE),
        "max_new_tokens": MAX_NEW_TOKENS,
        "patch_layers": PATCH_LAYERS,
        "attn_layers": ATTN_LAYERS,
        "examples": examples_out,
        "summary": summary,
    }

    out_path = OUTPUT_DIR / "generation_phase_probe.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {out_path}")

    for cond in ["baseline", "intervened"]:
        if sum(attn_count[cond]["pre_final"].values()) > 0:
            plot_attention_bars(
                summary[cond]["attn_pre_final"],
                f"Pre-Final Attention by Category ({cond})",
                f"generation_phase_attention_pre_final_{cond}.png",
            )
        if sum(attn_count[cond]["pre_answer"].values()) > 0:
            plot_attention_bars(
                summary[cond]["attn_pre_answer"],
                f"Pre-Answer Attention by Category ({cond})",
                f"generation_phase_attention_pre_answer_{cond}.png",
            )

        plot_attention_bars(
            summary[cond]["attn_end"],
            f"End-Token Attention by Category ({cond})",
            f"generation_phase_attention_end_{cond}.png",
        )

        if summary[cond]["logit_lens_pre_answer"]:
            plot_logit(
                summary[cond]["logit_lens_pre_answer"],
                f"Logit Lens Pre-Answer Rank ({cond})",
                f"generation_phase_logit_rank_pre_answer_{cond}.png",
                "avg_rank",
            )
            plot_logit(
                summary[cond]["logit_lens_pre_answer"],
                f"Logit Lens Pre-Answer LogP ({cond})",
                f"generation_phase_logit_logp_pre_answer_{cond}.png",
                "avg_logp",
            )

        plot_logit(
            summary[cond]["logit_lens_end"],
            f"Logit Lens End-Token Rank ({cond})",
            f"generation_phase_logit_rank_end_{cond}.png",
            "avg_rank",
        )
        plot_logit(
            summary[cond]["logit_lens_end"],
            f"Logit Lens End-Token LogP ({cond})",
            f"generation_phase_logit_logp_end_{cond}.png",
            "avg_logp",
        )

        for layer in ATTN_LAYERS:
            if sum(attn_count[cond]["pre_answer"].values()) > 0:
                matrix = np.array(summary[cond]["head_by_phase"]["pre_answer"][str(layer)]["mean_mass"])
                plot_head_heatmap(
                    matrix,
                    f"Head Attention by Category (Layer {layer}, pre-answer, {cond})",
                    f"generation_phase_headmap_pre_answer_L{layer}_{cond}.png",
                )
            matrix_end = np.array(summary[cond]["head_by_phase"]["end"][str(layer)]["mean_mass"])
            plot_head_heatmap(
                matrix_end,
                f"Head Attention by Category (Layer {layer}, end, {cond})",
                f"generation_phase_headmap_end_L{layer}_{cond}.png",
            )


if __name__ == "__main__":
    t.set_grad_enabled(False)
    main()
