# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
import numpy as np
from sklearn.linear_model import LogisticRegression
from ruptures.metrics import randindex, hausdorff
from typing import List, Tuple

# 15 colorblind-friendly colors
COLORS = ["#0072B2", "#009E73", "#D55E00", "#CC79A7", "#F0E442",
            "#56B4E9", "#E69F00", "#000000", "#0072B2", "#009E73",
            "#D55E00", "#CC79A7", "#F0E442", "#56B4E9", "#E69F00"]


def get_roc_metrics(real_preds, sample_preds):
    fpr, tpr, _ = roc_curve([0] * len(real_preds) + [1] * len(sample_preds), real_preds + sample_preds)
    roc_auc = auc(fpr, tpr)
    if roc_auc < 0.5:
        fpr, tpr, _ = roc_curve([1] * len(real_preds) + [0] * len(sample_preds), real_preds + sample_preds)
        roc_auc = auc(fpr, tpr)
    return fpr.tolist(), tpr.tolist(), float(roc_auc)

def get_roc_metrics_multi(real_preds, revise_preds, sample_preds):
    label = [0] * len(real_preds) + [1] * len(revise_preds) + [2] * len(sample_preds)
    preds = np.array(real_preds + revise_preds + sample_preds)
    if preds.ndim == 1:
        preds = preds.reshape(-1, 1)
    preds = LogisticRegression(random_state=0).fit(preds, label).predict_proba(preds)
    label = label_binarize(label, classes=[0, 1, 2])
    roc_auc = roc_auc_score(label, preds, multi_class='ovo', average='macro')
    return float(roc_auc)

def get_precision_recall_metrics(real_preds, sample_preds):
    precision, recall, _ = precision_recall_curve([0] * len(real_preds) + [1] * len(sample_preds),
                                                  real_preds + sample_preds)
    pr_auc = auc(recall, precision)
    if pr_auc < 0.5:
        precision, recall, _ = precision_recall_curve([1] * len(real_preds) + [0] * len(sample_preds),
                                                      real_preds + sample_preds)
        pr_auc = auc(recall, precision)
    return precision.tolist(), recall.tolist(), float(pr_auc)

def get_precision_recall_metrics_multi(real_preds, revise_preds, sample_preds):
    precision, recall, _ = precision_recall_curve([0] * len(real_preds) + [1] * len(revise_preds) + [2] * len(sample_preds), real_preds + revise_preds + sample_preds)
    pr_auc = auc(recall, precision)
    return precision.tolist(), recall.tolist(), float(pr_auc)

def get_rejection_rate(p_values, alpha=0.05):
    return float(np.mean(np.array(p_values) < alpha))

def get_cp_detection_metrics(true_cps, est_cps):
    ri = randindex(true_cps, est_cps)
    hau = hausdorff(true_cps, est_cps)
    return ri, hau

def get_hausdorff_tokenwise(true_cps, est_cps, ntokens_list):
    cumulative_tokens = np.cumsum(ntokens_list)
    token_true_cps = [cumulative_tokens[cp-1] for cp in true_cps]
    token_est_cps = [cumulative_tokens[cp-1] for cp in est_cps]
    hau_tokens = hausdorff(token_true_cps, token_est_cps)
    return hau_tokens / cumulative_tokens[-1]

def get_best_threshold_accuracy(y_true, y_score):
    """
    Compute best accuracy under thresholding, correctly handling ties.

    Parameters
    ----------
    y_true : array-like of shape (n,)
        Binary labels (0/1).
    y_score : array-like of shape (n,)
        Prediction scores (may contain ties).

    Returns
    -------
    best_acc : float
        Maximum achievable accuracy.
    best_thr : float
        Threshold achieving the maximum accuracy.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    # sort by score (ascending)
    order = np.argsort(y_score)
    y_true = y_true[order]
    y_score = y_score[order]

    # unique score blocks
    unique_scores, indices = np.unique(y_score, return_index=True)

    n = len(y_true)
    total_pos = y_true.sum()
    total_neg = n - total_pos

    # start with threshold < min(score): predict all 1
    tp = total_pos
    fp = total_neg
    tn = 0
    fn = 0

    best_acc = (tp + tn) / n
    best_thr = unique_scores[0] - 1e-12

    # sweep block by block
    for i in range(len(unique_scores)):
        start = indices[i]
        end = indices[i + 1] if i + 1 < len(indices) else n

        # move this entire block from positive to negative
        block_labels = y_true[start:end]
        tp -= block_labels.sum()
        fn += block_labels.sum()
        fp -= (end - start - block_labels.sum())
        tn += (end - start - block_labels.sum())

        acc = (tp + tn) / n

        # threshold between blocks
        if i + 1 < len(unique_scores):
            thr = 0.5 * (unique_scores[i] + unique_scores[i + 1])
        else:
            thr = unique_scores[i] + 1e-12

        if acc > best_acc:
            best_acc = acc
            best_thr = thr

    return best_acc, best_thr

def covering_metric(labels: List[int], est_cp: List[int]) -> float:
    """
    Compute covering metric C(G, G') following Arbeláez et al. (2010).

    Parameters
    ----------
    labels : list[int]
        Ground-truth labels (length T).
    est_cp : list[int]
        Estimated change points.

    Returns
    -------
    float
        Covering score.
    """
    T = len(labels)

    if len(est_cp) == 0:
        return 0.0

    G = segments_from_labels(labels)
    Gp = segments_from_cp(est_cp, T)

    score = []
    for A in G:
        len_A = A[1] - A[0]
        best_jaccard = max(jaccard(A, Ap) for Ap in Gp)
        score.append(len_A * best_jaccard)
    
    score = sum(score) / len(score)
    return score

def jaccard(seg1: Tuple[int, int], seg2: Tuple[int, int]) -> float:
    """
    Jaccard index between two intervals [s1,e1), [s2,e2)
    """
    s1, e1 = seg1
    s2, e2 = seg2

    inter = max(0, min(e1, e2) - max(s1, s2))
    union = (e1 - s1) + (e2 - s2) - inter

    return inter / union if union > 0 else 0.0

def segments_from_labels(labels: List[int]) -> List[Tuple[int, int]]:
    """
    Convert label sequence into contiguous segments.
    Returns half-open intervals [start, end).
    """
    segments = []
    start = 0
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            segments.append((start, i))
            start = i
    segments.append((start, len(labels)))
    return segments

def segments_from_cp(est_cp: List[int], T: int) -> List[Tuple[int, int]]:
    """
    Convert change points into segments.
    est_cp are assumed to be 0-based, segment boundaries.
    """
    cp = sorted([c for c in est_cp if 0 < c < T])
    boundaries = [0] + cp + [T]
    return [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]

if __name__ == "__main__":
    # synthetic example
    y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    y_score = np.array([2.4115355014801025, 2.4115355014801025, 2.4115355014801025, 2.4115355014801025, -2.2309446334838867, -2.2309446334838867, -2.2309446334838867, -2.2309446334838867, -2.2309446334838867, -2.2309446334838867])

    acc, thr = get_best_threshold_accuracy(y_true, y_score)

    print("Best accuracy:", acc)
    print("Best threshold:", thr)

    y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    est_cp = [5]
    cover = covering_metric(y_true, est_cp)
    print("Covering metric:", cover)
