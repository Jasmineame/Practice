import numpy as np
import ruptures as rpt
from metrics import get_best_threshold_accuracy, get_cp_detection_metrics, covering_metric
from sklearn.metrics import accuracy_score

def eval_score(results, thres=None):
    n_samples = len(results)
    half_samples = n_samples >> 1
    if thres is None:
        half_results_1 = results[:half_samples]
        half_results_2 = results[half_samples:len(results)]
        _, best_thres1 = get_best_threshold_accuracy([y for x in half_results_1 for y in x['labels']], [y for x in half_results_1 for y in x['predictions']])
        _, best_thres2 = get_best_threshold_accuracy([y for x in half_results_2 for y in x['labels']], [y for x in half_results_2 for y in x['predictions']])
    else:
        best_thres1 = thres
        best_thres2 = thres
    new_results = []
    for i in range(n_samples):
        if i < half_samples:
            best_thres = best_thres2
        else:
            best_thres = best_thres1
        num_sentence = len(results[i]['labels'])
        est_label_list = [(x > best_thres).astype(int) for x in results[i]['predictions']]
        acc = accuracy_score(results[i]['labels'], est_label_list)
        true_cp, est_cp = find_change_points(results[i]['labels']), find_change_points(est_label_list)
        eval_true_cp = true_cp + [num_sentence] if len(true_cp) > 0 else [0] + [num_sentence]
        eval_est_cp = est_cp + [num_sentence] if len(est_cp) > 0 else [0] + [num_sentence]
        ri, hau = get_cp_detection_metrics(eval_true_cp, eval_est_cp)
        cover = covering_metric(results[i]['labels'], est_label_list)
        if not isinstance(hau, float):
            hau = hau.item()
        if not isinstance(cover, float):
            cover = cover.item()
        if not isinstance(best_thres, float):
            best_thres = best_thres.item()
        new_results.append({
            "labels": results[i]['labels'], "predictions": results[i]['predictions'], "best_thres": best_thres, "true_cp": true_cp, "est_cp": est_cp,  
            "acc": acc, "rand": ri, "hausdorff": hau, 'cp_num_diff': len(true_cp) - len(est_cp), "covering": cover})
    return new_results

def eval_cp_accuracy(results):
    n_samples = len(results)
    half_samples = n_samples >> 1
    half_results_1 = results[:half_samples]
    half_results_2 = results[half_samples:len(results)]
    _, best_thres1 = get_best_threshold_accuracy([y for x in half_results_1 for y in x['labels']], [y for x in half_results_1 for y in x['predictions']])
    _, best_thres2 = get_best_threshold_accuracy([y for x in half_results_2 for y in x['labels']], [y for x in half_results_2 for y in x['predictions']])
    for i in range(n_samples):
        if i < half_samples:
            best_thres = best_thres2
        else:
            best_thres = best_thres1
        est_label_list = [(x > best_thres).astype(int) for x in results[i]['predictions']]
        acc = accuracy_score(results[i]['labels'], est_label_list)
        results[i]['best_thres'] = best_thres
        results[i]['acc'] = acc
    return results

def find_change_points(labels):
    """
    Find change points in a deterministic label sequence.

    Parameters
    ----------
    labels : list[str]
        Sequence of labels, e.g., ["L", "L", "H", "H", "L"]

    Returns
    -------
    cps : list[int]
        Change point indices (0-based), where a new segment starts.
    """
    cps = []
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            cps.append(i)
    return cps

def predict_sentence_cp(cp, sentence_list, llm_detector):
    '''
    predict whether a sentence is a change point based on the cp indices
    1. convert sentence_list into segments based on cp
    2. use model to predict each segment
    3. put the prediction results back to sentence level
    '''
    n = len(sentence_list)
    sentence_preds = [0] * n

    boundaries = [0] + cp + [n]

    # ------------------------------------------------
    # 2. segment-wise prediction
    # ------------------------------------------------
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]

        segment_sents = sentence_list[start:end]
        if len(segment_sents) == 0:
            continue
        score = llm_detector.score(" ".join(segment_sents))

        # ------------------------------------------------
        # 3. put results back to sentence level
        # ------------------------------------------------
        for j in range(start, end):
            sentence_preds[j] = score

    return sentence_preds

def max_cusum_statistic(prefix_w, prefix_wy, s_m, e_m, M, min_size=2):
    max_stats = np.empty(M, dtype=float)
    argmax_b = np.empty(M, dtype=int)
    for i in range(M):
        sm = int(s_m[i])
        em = int(e_m[i])

        L = em - sm
        if L < min_size:
            max_stats[i] = -np.inf
            argmax_b[i] = sm
            continue

        b = np.arange(sm, em + 1, dtype=int)

        W1 = prefix_w[b] - prefix_w[sm]      # sum_{t=sm+1}^b w_t
        W2 = prefix_w[em] - prefix_w[b]      # sum_{t=b+1}^{em} w_t
        WY1 = prefix_wy[b] - prefix_wy[sm]   # sum_{t=sm+1}^b w_t Y_t
        WY2 = prefix_wy[em] - prefix_wy[b]   # sum_{t=b+1}^{em} w_t Y_t

        valid = (W1 > 0) & (W2 > 0)
        stat = np.full(b.shape[0], -np.inf, dtype=float)
        if np.any(valid):
            mu1 = WY1[valid] / W1[valid]
            mu2 = WY2[valid] / W2[valid]
            scale = np.sqrt((W1[valid] * W2[valid]) / (W1[valid] + W2[valid]))
            stat[valid] = scale * np.abs(mu1 - mu2)

        j = int(np.argmax(stat))  # first maximizer if ties
        max_stats[i] = float(stat[j])
        argmax_b[i] = int(b[j])
    return max_stats, argmax_b

def NOT(Y, c_T, M, n_bkps, w=None, seed=None, min_size=4):
    """
    Python implementation of Algorithm 1 (NOT).

    Input
    -----
    Y   : array-like, shape (T,)
          data vector (Y_1, ..., Y_T)'
    c_T : float
          threshold
    M   : int
          tuning parameter (# random intervals per recursion)

    Output
    ------
    S : set of int
        estimated change-points S ⊂ {1, ..., T} (1-indexed)
    """
    if seed is not None:
        np.random.seed(seed)

    y0 = np.asarray(Y, dtype=float).reshape(-1)
    T = y0.size
    if T == 0:
        return set()
    if not isinstance(M, (int, np.integer)) or M < 0:
        raise ValueError("M must be a nonnegative integer.")
    if not np.isfinite(c_T):
        raise ValueError("c_T must be a finite number.")

    if w is None:
        w0 = np.ones(T, dtype=float)
    else:
        w0 = np.asarray(w, dtype=float).reshape(-1)
        if w0.size != T:
            raise ValueError("w must have the same length as Y.")
        if not np.all(np.isfinite(w0)):
            raise ValueError("w must be finite.")
        if np.any(w0 < 0):
            raise ValueError("w must be nonnegative (negative weights are not supported).")

    # Use 1-indexed convention to match the algorithm exactly.
    y = np.zeros(T + 1, dtype=float)
    y[1:] = y0
    ww = np.zeros(T + 1, dtype=float)
    ww[1:] = w0

    prefix_w = np.cumsum(ww)
    prefix_wy = np.cumsum(ww * y)  # prefix[t] = sum_{i=1}^t Y_i

    rng = np.random.default_rng()
    S = set()
    stack = [(1, T)]  # (s, e)

    while stack and (len(S) < n_bkps):
        s, e = stack.pop()

        # Step 2: if e - s <= 1 then STOP
        if e - s <= 1:
            continue

        # Step 5-7: draw M intervals; if M == 0 then STOP
        if M == 0:
            continue

        # Draw M pairs (u,v) uniformly from {s,...,e} with u != v, then sort -> (s_m, e_m)
        # and remove deduplicate intervals (𝓜 is a set in the algorithm)
        u = rng.integers(s, e + 1, size=M)
        v = rng.integers(s, e + 1, size=M)
        s_m = np.minimum(u, v).astype(int)
        e_m = np.maximum(u, v).astype(int)
        pairs = np.unique(np.stack([s_m, e_m], axis=1), axis=0)
        s_m = pairs[:, 0]
        e_m = pairs[:, 1]
        M_eff = pairs.shape[0]

        # Step 9 (i): compute max CUSUM statistic on each interval
        max_stats, argmax_b = max_cusum_statistic(prefix_w, prefix_wy, s_m, e_m, M_eff, min_size)

        # Step 9 (ii): define O as those exceeding c_T
        over = max_stats > c_T
        # Step 10-11: if O == ∅ then STOP
        if not np.any(over):
            continue

        # Step 13: m* = argmin_{m in O} (sum_{t=1}^{e_m} w_t - sum_{t=1}^{s_m} w_t)
        # i.e., minimal total weight on (s_m, e_m] when weights are one
        interval_w = (prefix_w[e_m] - prefix_w[s_m]).astype(float)
        interval_w_masked = np.where(over, interval_w, np.inf)
        m_star = np.where(interval_w_masked == np.min(interval_w_masked))[0]
        # interval_len = (e_m - s_m).astype(float)
        # interval_len_masked = np.where(over, interval_len, np.inf)
        # m_star = np.where(interval_len_masked == np.min(interval_len_masked))[0]
        
        m_star = m_star[np.argmax(max_stats[m_star])]

        # Step 14: b* := argmax ... on interval m*
        b_star = int(argmax_b[m_star])

        # Step 15: S := S ∪ {b*}
        if all([abs(s_tmp - b_star) >= min_size for s_tmp in S]):
            S.add(b_star)
            # Step 16-17: recurse on [s, b*] and [b*+1, e]
            stack.append((b_star + 1, e))
            stack.append((s, b_star))
        else:
            continue

    S = sorted(list(S))
    return S

def BinSeg(Y, n_bkps, w=None):
    if isinstance(Y, list):
        Y = np.array(Y)
    algo = rpt.Binseg(model="l2", min_size=1, jump=2).fit(Y, weight=w)
    S = algo.predict(n_bkps=n_bkps)
    return S[:-1]  # remove T

def DPSeg(Y, n_bkps, w=None):
    if isinstance(Y, list):
        Y = np.array(Y)
    algo = rpt.Dynp(model="l2", min_size=1, jump=2).fit(Y, weight=w)
    S = algo.predict(n_bkps=n_bkps)
    return S[:-1]  # remove T

if __name__ == "__main__":
    import matplotlib.pylab as plt

    n = 40  # number of samples
    n_bkps, sigma = 3, 1.0  # number of change points, noise standard deviation
    signal, bkps = rpt.pw_constant(n, 1, n_bkps, noise_std=sigma, seed=0)
    bkps = bkps[:-1]  # remove n

    # --- 2) run Binary Segmentation (fixed k=3) ---
    my_bkps = BinSeg(signal, n_bkps=n_bkps)

    # --- 3) run NOT ---
    c_T = 1   # threshold (tune this; smaller => more detections)
    M = 10000      # number of random intervals per recursion
    not_cps = NOT(signal, c_T=c_T, M=M, n_bkps=3)
    not_cps = sorted(not_cps)

    print("True change-points (1-indexed):", bkps)
    print("NOT estimated change-points:", not_cps)
    print(f"Binary segmentation (L2, k={n_bkps}) estimated change-points:", my_bkps)

    # --- 4) plot ---
    t = np.arange(1, len(signal) + 1)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(t, signal, label="Observed Y")

    for cp in bkps:
        ax.axvline(cp, color="black", linestyle="--", label="True change-point" if cp == bkps[0] else None)
    for cp in not_cps:
        ax.axvline(cp, color="red", linestyle=":", label="NOT estimate" if cp == not_cps[0] else None)
    for cp in my_bkps:
        ax.axvline(cp, color="green", linestyle="-.", label="BinSeg (L2) estimate" if cp == my_bkps[0] else None)

    ax.set_title("NOT vs Binary Segmentation on a 3-change-point synthetic signal")
    ax.set_xlabel("t")
    ax.set_ylabel("Y_t")
    ax.legend(loc="upper left", ncol=3, fontsize=9)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.show()