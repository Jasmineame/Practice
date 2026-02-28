import torch
from model import load_tokenizer, load_model

def get_sampling_discrepancy_analytic(logits_ref, logits_score, labels):
    assert logits_ref.shape[0] == 1
    assert logits_score.shape[0] == 1
    assert labels.shape[0] == 1
    if logits_ref.size(-1) != logits_score.size(-1):
        # print(f"WARNING: vocabulary size mismatch {logits_ref.size(-1)} vs {logits_score.size(-1)}.")
        vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
        logits_ref = logits_ref[:, :, :vocab_size]
        logits_score = logits_score[:, :, :vocab_size]

    labels = labels.unsqueeze(-1) if labels.ndim == logits_score.ndim - 1 else labels
    lprobs_score = torch.log_softmax(logits_score, dim=-1)
    probs_ref = torch.softmax(logits_ref, dim=-1)
    log_likelihood = lprobs_score.gather(dim=-1, index=labels).squeeze(-1)
    mean_ref = (probs_ref * lprobs_score).sum(dim=-1)
    var_ref = (probs_ref * torch.square(lprobs_score)).sum(dim=-1) - torch.square(mean_ref)
    discrepancy = (log_likelihood.sum(dim=-1) - mean_ref.sum(dim=-1)) / var_ref.sum(dim=-1).sqrt()
    discrepancy = discrepancy.mean()
    return discrepancy.item()

def get_fastdetectgpt_score(args, input_text,
    scoring_tokenizer=None, scoring_model=None, reference_tokenizer=None, reference_model=None):
    # load model
    if scoring_model is None:
        scoring_tokenizer = load_tokenizer(args.scoring_model_name, args.dataset, args.cache_dir)
        scoring_model = load_model(args.scoring_model_name, args.device, args.cache_dir)
    scoring_model.eval()

    if reference_model is None:
        if args.reference_model_name != args.scoring_model_name:
            reference_tokenizer = load_tokenizer(args.reference_model_name, args.dataset, args.cache_dir)
            reference_model = load_model(args.reference_model_name, args.device, args.cache_dir)
            reference_model.eval()

    # evaluate criterion
    name = "sampling_discrepancy_analytic"
    criterion_fn = get_sampling_discrepancy_analytic
    prob_estimator = ProbEstimator(args)

    text = input_text
    # evaluate text
    tokenized = scoring_tokenizer(text, return_tensors="pt", padding=True, return_token_type_ids=False).to(args.device)
    labels = tokenized.input_ids[:, 1:]
    with torch.no_grad():
        logits_score = scoring_model(**tokenized).logits[:, :-1]
        if args.reference_model_name == args.scoring_model_name:
            logits_ref = logits_score
        else:
            tokenized = reference_tokenizer(text, return_tensors="pt", padding=True, return_token_type_ids=False).to(args.device)
            assert torch.all(tokenized.input_ids[:, 1:] == labels), "Tokenizer is mismatch."
            logits_ref = reference_model(**tokenized).logits[:, :-1]
        crit = criterion_fn(logits_ref, logits_score, labels)
    # estimate the probability of machine generated text
    prob = prob_estimator.crit_to_prob(crit)
    return crit, prob
