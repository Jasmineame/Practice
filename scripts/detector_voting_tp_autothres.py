from pure_llm_detector import LLMDetector
import json
import argparse
import numpy as np
from utils import load_data, process_json_file_nested
from data_builder_sentence import count_tokens
from rewrite_machine import PrefixSampler
from utils_cp import eval_score, find_change_points
from metrics import get_cp_detection_metrics, covering_metric, get_best_threshold_accuracy, get_roc_metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score
from tqdm import tqdm
import sys
import torch
import pandas as pd

def get_regen_samples(sampler, text, regen_number=None, rewrite_model_name=None):
    if regen_number is None:
        data = [text] * sampler.args.regen_number
    else:
        data = [text] * regen_number
    data = sampler.generate_samples(data, batch_size=sampler.args.batch_size, rewrite_model_name=rewrite_model_name)
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
    parser.add_argument('--quantile', type=float, default=0.05, help="quantile for computing the statistics for thresholding")
    parser.add_argument('--batch_size', type=int, default=2)
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
    parser.add_argument('--local_files_only', type=bool, default=False, help="whether to only use local files for loading models (no downloading from HuggingFace)")
    ### parameter for oracle comparison
    parser.add_argument('--do_oracle', action='store_true', default=False, help="whether to compute oracle evaluation results (upper bound of the method)")
    args = parser.parse_args()
    if args.train_thres:
        print(f"DEBUG: 接收到的 thres_regen_number 是: {args.thres_regen_number}")
        name = f"voting_tp.{args.phi.lower()}.width{args.width}.thres-train.regen{args.thres_regen_number}.quantile{args.quantile}"
    elif args.do_oracle:
        name = f"voting_tp.{args.phi.lower()}.width{args.width}.oracle"

    # 0. load detection model
    llm_detector = LLMDetector(args)

    if args.train_thres:
        sampler = PrefixSampler(args)

    val_data = load_data(args.eval_dataset)
    val_data = process_json_file_nested(args.eval_dataset)
    n_samples = len(val_data['sampled_sentence'])
    if args.num_subsample > 0:
        n_samples = min(n_samples, args.num_subsample)

    results = []
    for i in tqdm(range(n_samples), desc="Processing samples"):
        sens_list = val_data['sampled_sentence'][i]
        sens_label_list = val_data['source_label'][i]
        label_map = {"L": 1, "H": 0}
        sens_label_list = [label_map[x] for x in sens_label_list]
        toks_label_list = val_data['token_labels'][i]
        num_sentence = len(sens_list)
        if num_sentence <= 1:
            continue

        ## compute null-hypothesis threshold scores from rewritten text
        if args.train_thres:
            rewrite_text = get_regen_samples(sampler, val_data['sampled'][i], args.thres_regen_number, args.rewrite_model_name)
            all_rewrite_score = []
            for idx in range(len(rewrite_text)):
                num_rewrite_toks, rewrite_toks_list = count_tokens(rewrite_text[idx])
                if num_rewrite_toks < args.width:
                    stat = llm_detector.score(" ".join(rewrite_toks_list)) # all rewrite tokens as input.
                    rewrite_scores = [stat] * num_rewrite_toks
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
        num_toks, sample_toks_list = len(val_data['tokens'][i]), val_data['tokens'][i]
        if num_toks < args.width:
            stat = llm_detector.score(" ".join(sample_toks_list))
            majority_vote_preds = [stat] * num_toks
        else:
            majority_vote_preds = [[] for _ in range(num_toks)]
            for window_start in range(0, max(1, num_toks-args.width+1), 1):
                prediction_score = llm_detector.score(" ".join(sample_toks_list[window_start:(window_start+args.width)]))
                for vote_idx in range(window_start, window_start+args.width):
                    majority_vote_preds[vote_idx].append(prediction_score)
            majority_vote_preds = [sum(sub_list)/len(sub_list) for sub_list in majority_vote_preds] 

        if args.do_oracle:
            best_acc, best_thres = get_best_threshold_accuracy(toks_label_list, majority_vote_preds)
            precision = precision_score(toks_label_list, [int(x > best_thres) for x in majority_vote_preds], zero_division=0)
            recall = recall_score(toks_label_list, [int(x > best_thres) for x in majority_vote_preds], zero_division=0)
        # else:
        #     best_acc, best_thres = None, None

        if args.train_thres:
            ## per-sample threshold from null-hypothesis rewritten scores
            thres_i = float(np.quantile(all_rewrite_score, args.quantile))
            est_label_list = [int(x > thres_i) for x in majority_vote_preds]
            _, _, auc = get_roc_metrics(toks_label_list, est_label_list)
            acc = accuracy_score(toks_label_list, est_label_list)
            precision = precision_score(toks_label_list, est_label_list, zero_division=0)
            recall = recall_score(toks_label_list, est_label_list, zero_division=0)
            results.append({
                "labels": toks_label_list, "predictions": majority_vote_preds,
                "null_scores": all_rewrite_score, "thres": thres_i,
                "acc": acc, "precision": precision, "recall": recall,
                "auc": auc,
            })
        else:
            results.append({"labels": toks_label_list, "predictions": majority_vote_preds,
                            "acc": best_acc, "thres": best_thres,
                            "precision": precision, "recall": recall})

    ## evaluate detection results (only needed for non-train_thres path)
    # if not args.train_thres:
    #     # new_results = eval_score(results)
    #     if args.do_oracle:
    #         for i in tqdm(range(n_samples), desc="Processing samples"):
    #             new_results[i]["acc_oracle"] = results[i]["acc_oracle"]
    #             new_results[i]["thres_oracle"] = results[i]["thres_oracle"]
    #     results = new_results

    # print evaluation results
    if args.train_thres:
        class_eval = {'acc': [x["acc"] for x in results],
                      'precision': [x["precision"] for x in results],
                      'recall': [x["recall"] for x in results]}
        metrics = {'acc': np.mean(class_eval['acc']).tolist(), 'acc_std': np.std(class_eval['acc']).tolist(),
                   'precision': np.mean(class_eval['precision']).tolist(), 'precision_std': np.std(class_eval['precision']).tolist(),
                   'recall': np.mean(class_eval['recall']).tolist(), 'recall_std': np.std(class_eval['recall']).tolist()}
    else:
        class_eval = {'acc': [x["acc"] for x in results],
                      'precision': [x["precision"] for x in results],
                      'recall': [x["recall"] for x in results]}
        metrics = {'acc': np.mean(class_eval['acc']).tolist(), 'acc_std': np.std(class_eval['acc']).tolist(),
                   'precision': np.mean(class_eval['precision']).tolist(), 'precision_std': np.std(class_eval['precision']).tolist(),
                   'recall': np.mean(class_eval['recall']).tolist(), 'recall_std': np.std(class_eval['recall']).tolist()}
    # print(f"Oracle Accuracy (mean/std): {np.mean(class_eval['acc_oracle']):.2f}/{np.std(class_eval['acc_oracle']):.2f}", 
    # )
    results_file = f'{args.output_file}.{name}.json'
    results = { 
        'name': f'{name}',
        'info': {'n_samples': n_samples},
        'metrics': metrics,
        'raw_results': results,
    }
    with open(results_file, 'w') as fout:
        json.dump(results, fout, indent=2)
        print(f'Results written into {results_file}')