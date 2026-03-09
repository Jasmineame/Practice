from collections import defaultdict
import json
import torch
from spacy.lang.en import English
from typing import Dict, Any, List, Tuple, Optional
import re
import glob
from pathlib import Path

import matplotlib.pyplot as plt

nlp = English()
nlp.add_pipe("sentencizer")  # rule-based, no parser

nlp_tp = English()

def count_sentences(input_article):
    doc = nlp(input_article)
    sentences = [sent.text.strip() for sent in doc.sents]
    return len(sentences), sentences

def count_tokens(input_article):
    doc = nlp_tp(input_article)
    tokens = [token.text for token in doc if not token.is_space]
    return len(tokens), tokens

def tokenize_and_label_nested(
    example: Dict[str, Any],
    sentences_key: str = "sampled_sentence",  # List[List[str]]
    label_key: str = "source_label",           # List[List[int]]
) -> Dict[str, Any]:
    """
    输入 example:
      {
        "sampled_sentences": [
            [s00, s01, ...],   # 第0个 sample text 的句子列表
            [s10, s11, ...],   # 第1个 sample text 的句子列表
            ...
        ],
        "source_label": [
            [l00, l01, ...],   # 与上面对齐（逐句 0/1）
            [l10, l11, ...],
            ...
        ]
      }

    输出（同样两层）:
      - tokens:       List[List[str]]  # 每个 sample text 一条 token 序列（已把多句拼起来）
      - token_labels: List[List[int]]  # 与 tokens 对齐
    """
    nested_sentences: List[List[str]] = example[sentences_key]
    nested_labels: List[List[int]] = example[label_key]

    if len(nested_sentences) != len(nested_labels):
        raise ValueError(
            f"Outer length mismatch: {sentences_key}={len(nested_sentences)} vs {label_key}={len(nested_labels)}"
        )

    all_tokens: List[List[str]] = []
    all_token_labels: List[List[int]] = []

    for t_idx, (sent_list, lab_list) in enumerate(zip(nested_sentences, nested_labels)):
        if len(sent_list) != len(lab_list):
            raise ValueError(
                f"Inner length mismatch at text {t_idx}: "
                f"{sentences_key}[{t_idx}]={len(sent_list)} vs {label_key}[{t_idx}]={len(lab_list)}"
            )

        tokens_one: List[str] = []
        labels_one: List[int] = []

        for s_idx, (sent, lab) in enumerate(zip(sent_list, lab_list)):
            lab = 1 if lab == "L" else 0
            _, sent_tokens = count_tokens(sent)

            tokens_one.extend(sent_tokens)
            labels_one.extend([lab] * len(sent_tokens))

        all_tokens.append(tokens_one)
        all_token_labels.append(labels_one)

    return {
        "tokens": all_tokens,
        "token_labels": all_token_labels,
    }

def process_json_file_nested(
    input_path: str,
    output_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    f = f"{input_path}.raw_data.json"
    with open(f, "r", encoding="utf-8") as f:
        data = json.load(f)

    processed = []
    out = tokenize_and_label_nested(
        data,
    )
    processed.append({**data, **out})

    if output_path is not None:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(processed[0], f, ensure_ascii=False)

    return processed[0]

class GpuMem:
    def __init__(self, device=0):
        self.device = device
        self.peak = None

    def __enter__(self):
        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        self.before = torch.cuda.memory_allocated(self.device)
        return self

    def __exit__(self, exc_type, exc, tb):
        torch.cuda.synchronize(self.device)
        self.peak = torch.cuda.max_memory_allocated(self.device)
    
    def memory_usage(self):
        '''get memory usage (measured in GB)'''
        unit = 1e9
        return (self.peak - self.before) / unit

def load_data(input_file):
    data_file = f"{input_file}.raw_data.json"
    with open(data_file, "r") as fin:
        data = json.load(fin)
        print(f"Raw data loaded from {data_file}")
    return data

def merge_dicts_of_lists(dataset_list) -> dict:
    """
    将一系列 dict(键→list) 合并为一个 dict，
    同一个键对应的 list 会被 extend 到一起。
    """
    merged = defaultdict(list)
    for d in dataset_list:
        for key, value in d.items():
            # 如果 value 本身是 list，则 extend；否则 append
            if isinstance(value, list):
                merged[key].extend(value)
            else:
                merged[key].append(value)
    return dict(merged)

def load_training_data(train_dataset_list):
    dataset_list = []
    for data_name in train_dataset_list:
        dataset = load_data(data_name)
        dataset_list.append(dataset)
    ## combine training data
    train_data = merge_dicts_of_lists(dataset_list)
    return train_data

def load_rewrite_data(rewrite_data_list):
    dataset_list = []
    for data_name in rewrite_data_list:
        data_file = f"{data_name}.json"
        with open(data_file, "r") as fin:
            dataset = json.load(fin)
            print(f"Raw rewrite data loaded from {data_file}")
        dataset = {
            'rewrite_original': [x['rewrite_original'] for x in dataset], 
            'rewrite_sampled': [x['rewrite_sampled'] for x in dataset]
        }
        dataset_list.append(dataset)
    train_rewrite_data = merge_dicts_of_lists(dataset_list)
    return train_rewrite_data

def load_training_data2(train_dataset_list, base_dir):
    dataset_list = []
    for data_name in train_dataset_list:
        data_file = f'{base_dir}/{data_name}'
        with open(data_file, "r") as fin:
            data = json.load(fin)
            print(f"Raw data loaded from {data_file}")
        dataset_list.append(data)
    ## combine training data
    train_data = merge_dicts_of_lists(dataset_list)
    return train_data

def separated_string(s: str):
    '''
    return a list of strings from a string
    '''
    return s.split('&')

def load_acc(json_path: Path) -> float:
    """Load metrics.acc (mean) from a single JSON file."""
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # ── Handle common JSON structures ─────────────────────────────
    # Structure 1: {"metrics": {"acc_oracle": 0.85, ...}, ...}
    if isinstance(data, dict) and "metrics" in data:
        return float(data["metrics"]["acc"])

    # Structure 2: [{"metrics": {...}}, ...]  → average over all records
    if isinstance(data, list):
        vals = []
        for r in data:
            if isinstance(r, dict) and "metrics" in r:
                vals.append(float(r["metrics"]["acc"]))
        if vals:
            return sum(vals) / len(vals)

    raise ValueError(
        f"Cannot find metrics.acc in {json_path.name}. "
        "Please check the JSON structure and update load_acc() accordingly."
    )

def collect_data(data_dir: str, pattern: str):
    files = sorted(Path(data_dir).glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No files matching {pattern!r} found in {data_dir!r}. "
            "Please check DATA_DIR and FILE_GLOB."
        )

    pat = re.compile(r"\.width(\d+)\.")   # extract width number from filename

    width_error = {}
    for fp in files:
        m = pat.search(fp.name)
        if not m:
            print(f"[skip] Cannot extract width from filename: {fp.name}")
            continue
        width = int(m.group(1))
        acc   = load_acc(fp)
        error = 1.0 - acc
        print(f"  width={width:>4d}  acc={acc:.4f}  error={error:.4f}  ({fp.name})")
        width_error[width] = error

    widths = sorted(width_error)
    errors = [width_error[w] for w in widths]
    return widths, errors


def plot(widths, errors, title=None, save_path=None):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    ax.plot(widths, errors, marker="o", linewidth=2, markersize=7)

    # ── Axes ──────────────────────────────────────────────────────
    ax.set_xlabel("Width", fontsize=12)
    ax.set_ylabel("Classification Error  (1 − mean acc)", fontsize=12)
    ax.set_xscale("log", base=2)          # log scale suits powers-of-2 widths
    ax.set_xticks(widths)
    ax.set_xticklabels([str(w) for w in widths])
    ax.grid(True, which="both", linestyle="--", alpha=0.4)

    if title:
        ax.set_title(title, fontsize=11, pad=10)

    # ── Annotate each data point with its value ───────────────────
    for w, e in zip(widths, errors):
        ax.annotate(
            f"{e:.3f}",
            xy=(w, e),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200)
        print(f"\nFigure saved to {save_path}")
    plt.show()


if __name__ == "__main__":

    # ─────────────────────────────────────────────
    # Configuration: edit these as needed
    # ─────────────────────────────────────────────
    DATA_DIR  = "/root/project/Practice/code/exp_location_single_cp/results"           # directory containing the JSON files
    FILE_GLOB = "writing_gemma-2b-instruct_rewrite.voting_tp.ft.width*.thres-train.regen1.quantile0.05.json"
    SAVE_PATH = "/root/project/Practice/code/exp_location_single_cp/results/plot/width_vs_error_thres-train.png"   # set to None to display only (no save)
    TITLE     = "Width vs Classification Error\n(gemma-2b-instruct / voting_tp / ft)"
    # ─────────────────────────────────────────────

    print(f"Scanning directory : {Path(DATA_DIR).resolve()}")
    print(f"File pattern       : {FILE_GLOB}\n")

    widths, errors = collect_data(DATA_DIR, FILE_GLOB)
    plot(widths, errors, title=TITLE, save_path=SAVE_PATH)

