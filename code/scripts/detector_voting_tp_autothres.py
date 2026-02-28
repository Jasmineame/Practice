from pure_llm_detector import LLMDetector
import json
import argparse
import numpy as np
from utils import load_data
from data_builder_sentence import count_tokens
from rewrite_machine import PrefixSampler
from utils_cp import eval_score, find_change_points
from metrics import get_cp_detection_metrics, covering_metric
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import sys
import torch
import pandas as pd


def get_regen_samples(sampler, text, regen_number=None):
    if regen_number is None:
        data = [text] * sampler.args.regen_number
    else:
        data = [text] * regen_number
    data = sampler.generate_samples(data, batch_size=sampler.args.batch_size)
    return data['sampled']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ### parameter for detection model
    parser.add_argument('--base_model', type=str, default="gemma-1b")
    parser.add_argument('--aux_model', type=str, default="gemma-1b-instruct")
    parser.add_argument('--from_pretrained', type=str, default="scripts/FineTune/ckpt/")
    parser.add_argument('--phi', type=str, default="Bino", help="pure LLM-generated detection method: FT (fine-tuning method) or Bino (binoculars) ", choices=["FT", "Bino", "FDGPT"])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--width', type=int, default=3, help="window width for voting")
    parser.add_argument('--num_subsample', type=int, default=-1, help="number of samples for evaluation, -1 means all samples")
    ### parameter for computing the threshold
    parser.add_argument('--train_thres', action='store_true', help="whether to train the threshold")
    parser.add_argument('--rewrite_model_name', type=str, default="mistralai/Ministral-8B-Instruct-2410", help="the model for rewriting the input so as to select the threshold")
    parser.add_argument('--thres_regen_number', type=int, default=1, help="rewrite number for each input")
    parser.add_argument('--quantile', type=float, default=0.95, help="quantile for computing the statistics for thresholding")
    parser.add_argument('--batch_size', type=int, default=1)
    ### parameter for rewriting text
    parser.add_argument('--max_new_tokens', type=int, default=1000)
    parser.add_argument('--do_top_k', action='store_true')
    parser.add_argument('--top_k', type=int, default=40)
    parser.add_argument('--do_top_p', action='store_true')
    parser.add_argument('--top_p', type=float, default=0.96)
    parser.add_argument('--do_temperature', action='store_true')
    parser.add_argument('--temperature', type=float, default=0.8)
    ### parameter for computing details
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--eval_dataset', type=str, default="./exp_location/data/squad_mistralai-8b-instruct_rewrite")
    parser.add_argument('--output_file', type=str, default="./exp_location/results/squad_mistralai-8b-instruct_rewrite")
    parser.add_argument('--cache_dir', type=str, default="../cache")
    args = parser.parse_args()
    name = f"voting_sp.{args.phi.lower()}"

    print(f"DEBUG: 接收到的 thres_regen_number 是: {args.thres_regen_number}")

    # 0. load detection model
    llm_detector = LLMDetector(args)

    if args.train_thres:
        sampler = PrefixSampler(args)

    val_data = load_data(args.eval_dataset)
    n_samples = len(val_data['sampled_sentence'])
    if args.num_subsample > 0:
        n_samples = min(n_samples, args.num_subsample)

    results = []
    for i in tqdm(range(n_samples), desc="Processing samples"):
        sens_list = val_data['sampled_sentence'][i]
        sens_label_list = val_data['source_label'][i]
        label_map = {"L": 1, "H": 0}
        sens_label_list = [label_map[x] for x in sens_label_list]
        num_sentence = len(sens_list)
        if num_sentence <= 1:
            continue

        ## compute null-hypothesis threshold scores from rewritten text
        if args.train_thres:
            rewrite_text = get_regen_samples(sampler, val_data['sampled'][i], args.thres_regen_number)
            all_rewrite_score = []
            for idx in range(len(rewrite_text)):
                num_rewrite_toks, rewrite_toks_list = count_tokens(rewrite_text[idx])
                if num_rewrite_toks < args.width:
                    rewrite_scores = []
                    for j in range(num_rewrite_toks):
                        stat = llm_detector.score(rewrite_toks_list[j])
                        rewrite_scores.append(stat)
                else:
                    rewrite_votes = [[] for _ in range(num_rewrite_toks)]
                    for window_start in range(0, max(1, num_rewrite_toks - args.width + 1), 1):
                        with torch.no_grad():
                            prediction_score = llm_detector.score(" ".join(rewrite_toks_list[window_start:(window_start + args.width)]))
                        for vote_idx in range(window_start, window_start + args.width):
                            rewrite_votes[vote_idx].append(prediction_score)
                    rewrite_scores = [sum(sub_list) / len(sub_list) for sub_list in rewrite_votes]
                all_rewrite_score.extend(rewrite_scores)
            # rewrite_scores = pd.DataFrame(all_rewrite_score).mean(axis=0).tolist()  # 对多个rewrite的结果取平均，得到每个sentence的rewrite score
        ## segment-level statistics computation
        if num_sentence < args.width:
            majority_vote_preds = []
            for j in range(num_sentence):
                start, end = j, j+1
                stat = llm_detector.score(sens_list[start:end][0])
                majority_vote_preds.append(stat)
        else:
            majority_vote_preds = [[] for _ in range(num_sentence)]
            for window_start in range(0, max(1, len(sens_list)-args.width+1), 1):
                prediction_score = llm_detector.score(" ".join(sens_list[window_start:(window_start+args.width)]))
                for vote_idx in range(window_start, window_start+args.width):
                    majority_vote_preds[vote_idx].append(prediction_score)
            majority_vote_preds = [sum(sub_list)/len(sub_list) for sub_list in majority_vote_preds]

        if args.train_thres:
            ## per-sample threshold from null-hypothesis rewritten scores
            thres_i = float(np.quantile(rewrite_scores, args.quantile))
            est_label_list = [int(x > thres_i) for x in majority_vote_preds]
            acc = accuracy_score(sens_label_list, est_label_list)
            true_cp = find_change_points(sens_label_list)
            est_cp = find_change_points(est_label_list)
            eval_true_cp = true_cp + [num_sentence] if len(true_cp) > 0 else [0] + [num_sentence]
            eval_est_cp = est_cp + [num_sentence] if len(est_cp) > 0 else [0] + [num_sentence]
            ri, hau = get_cp_detection_metrics(eval_true_cp, eval_est_cp)
            cover = covering_metric(sens_label_list, est_label_list)
            if not isinstance(hau, float): hau = hau.item()
            if not isinstance(cover, float): cover = cover.item()
            results.append({
                "labels": sens_label_list, "predictions": majority_vote_preds,
                "null_scores": rewrite_scores, "best_thres": thres_i,
                "true_cp": true_cp, "est_cp": est_cp,
                "acc": acc, "rand": ri, "hausdorff": hau,
                "cp_num_diff": len(true_cp) - len(est_cp), "covering": cover
            })
        else:
            results.append({"labels": sens_label_list, "predictions": majority_vote_preds})

    ## evaluate detection results (only needed for non-train_thres path)
    if not args.train_thres:
        results = eval_score(results)

    # print evaluation results
    class_eval = {'acc': [x["acc"] for x in results], 'rand': [x["rand"] for x in results], 'hausdorff': [x["hausdorff"] for x in results], 'cp_num_diff': [x["cp_num_diff"] for x in results], 'covering': [x["covering"] for x in results]}
    print(
        f"Best Accuracy (mean/std): {np.mean(class_eval['acc']):.2f}/{np.std(class_eval['acc']):.2f}", 
        f"Rand Index (mean/std): {np.mean(class_eval['rand']):.2f}/{np.std(class_eval['rand']):.2f}", 
        f"Hausdorff Distance (mean/std): {np.mean(class_eval['hausdorff']):.2f}/{np.std(class_eval['hausdorff']):.2f}", 
        f"CP Number Difference (mean/std): {np.mean(class_eval['cp_num_diff']):.2f}/{np.std(class_eval['cp_num_diff']):.2f}",
        f"Covering Metric (mean/std): {np.mean(class_eval['covering']):.2f}/{np.std(class_eval['covering']):.2f}",
    )
    # results
    results_file = f'{args.output_file}.{name}.json'
    results = { 
        'name': f'{name}',
        'info': {'n_samples': n_samples},
        'metrics': {
            'acc': np.mean(class_eval['acc']).tolist(), 'acc_std': np.std(class_eval['acc']).tolist(),
            'rand': np.mean(class_eval['rand']).tolist(), 'rand_std': np.std(class_eval['rand']).tolist(), 
            'hausdorff': np.mean(class_eval['hausdorff']).tolist(), 'hausdorff_std': np.std(class_eval['hausdorff']).tolist(),
            'cp_num_diff': np.mean(class_eval['cp_num_diff']).tolist(), 'cp_num_diff_std': np.std(class_eval['cp_num_diff']).tolist(),
            'covering': np.mean(class_eval['covering']).tolist(), 'covering_std': np.std(class_eval['covering']).tolist(),
        },
        'raw_results': results,
    }
    with open(results_file, 'w') as fout:
        json.dump(results, fout, indent=2)
        print(f'Results written into {results_file}')