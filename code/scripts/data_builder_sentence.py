import argparse
import os
import numpy as np
import random
import json
import tqdm
import time
import re
import custom_datasets
from utils import count_sentences, count_tokens
import huggingface_hub

huggingface_hub.login("")

from model import load_model, load_tokenizer
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def deduce_longer_segment(num_sentence, original_sentence, n_split):
    num_sentence_split = int(num_sentence / n_split)
    if num_sentence_split > 16:
        original_sentence = original_sentence[:(16 * n_split)]
        num_sentence = len(original_sentence)
    return num_sentence, original_sentence

# def count_sentences(text: str):
#     """
#     统计文本中的句子数量，并保留句尾标点。
#     句子由 ., ?, !, 。, ？, ！ 分隔。
#     """
#     # 使用正则匹配：非分隔符+分隔符
#     sentences = re.findall(r'[^.!?。！？]+[.!?。！？]?', text)
#     # 去掉首尾空白
#     sentences = [s.strip() for s in sentences if s.strip()]
#     return len(sentences), sentences

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

def load_data(args, dataset, key):
    # strip newlines from each example; replace one or more newlines with a single space
    def _strip_newlines(text):
        return ' '.join(text.split())

    # load data
    if dataset in custom_datasets.DATASETS:
        data = custom_datasets.load(dataset, args.cache_dir)
    else:
        data = custom_datasets.load_dataset(dataset, split='train', cache_dir=args.cache_dir)[key]

    # get unique examples, strip whitespace, and remove newlines
    # then take just the long examples, shuffle, take the first 5,000 to tokenize to save time
    # then take just the examples that are <= 512 tokens (for the base model)
    # then generate n_samples samples

    # remove duplicates from the data
    data = list(dict.fromkeys(data))  # deterministic, as opposed to set()

    # strip whitespace around each example
    data = [x.strip() for x in data]

    # remove newlines from each example
    data = [_strip_newlines(x) for x in data]

    # try to keep only examples with > 250 words
    if dataset in ['writing', 'squad', 'xsum', 'yelp_polarity', "essay"]:
        long_data = [x for x in data if len(x.split()) > 800]
        if len(long_data) > 0:
            data = long_data

    data = data[:5_000]

    return data

def llama_filter(messages, generation_args, pipe, output):
    attempt = 0
    while "Can I help you with something else?" in output or output.startswith("I cannot"):
        attempt += 1
        print(f"Can not generate content... Retrying [Attempt {attempt}]")
        output = pipe(messages, **generation_args)
        output = output[0]['generated_text']
        if attempt==15:
            print("Failed to rewrite")
            break
    if output.startswith("Here is a") or output.startswith("Here's a"):
        output = re.sub(r'Here.*?:', '', output, count=1)
    output = output.replace("\n\n","")
    return output


def _split_sentence_indices(num_sentences: int, k: int, split_type: str):
    """Return a list of (start, end) indices that partitions [0, num_sentences) into k segments.

    - equal_len: segments have as equal sentence counts as possible.
    - random: random cut points; segments may be unbalanced.
    """
    if num_sentences < 0:
        raise ValueError(f"num_sentences must be >= 0, got {num_sentences}")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if num_sentences == 0:
        return [(0, 0)]

    k = min(k, num_sentences)  # avoid empty segments

    # if split_type == 'random' and k > 1:
    #     # Choose k-1 cut points from 1..num_sentences-1
    #     cut_points = np.random.choice(np.arange(1, num_sentences), size=k - 1, replace=False)
    #     cut_points = sorted(cut_points.tolist())
    #     boundaries = [0] + cut_points + [num_sentences]
    #     return [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]

    # Default: equal_len
    base = num_sentences // k
    remainder = num_sentences % k

    blocks = []
    start = 0
    for i in range(k):
        # 前 remainder 个 block 多 1 个
        block_size = base + (1 if i < remainder else 0)
        end = start + block_size
        if block_size > 0:
            blocks.append((start, end))
        start = end
        
    return blocks


def _clean_llm_prefix(text: str) -> str:
    # Remove occasional "Here is/Here's ...:" prefaces
    cleaned = re.sub(r"(?i)^\s*here(?:'s| is)\s+[^:：]+[:：]\s*", "", text.strip())
    return cleaned.strip()

def openai_sampler(original_texts, args):
    from openai import OpenAI
    client = OpenAI()
    n_samples = len(original_texts)

    # kwargs = {"max_tokens": 500}
    kwargs = {"model": 'gpt-4o'}
    if args.do_top_p:
        kwargs['top_p'] = args.top_p
    elif args.do_top_k:
        kwargs['top_k'] = args.top_k
    elif args.do_temperature:
        kwargs['temperature'] = args.temperature

    system_prompt = 'You are a professional rewriting expert and you can help paraphrasing in English without missing the original details. Please ensure that, the rewritten text has the same number of sentence as the original text.'
    user_prompts = ['Please rewrite:'] * n_samples

    sampled_sentence_list = []
    source_label_list = []
    response_list =[]
    for idx in tqdm.tqdm(range(n_samples)):
        num_sentence, original_sentence = count_sentences(original_texts[idx])
        num_sentence, original_sentence = deduce_longer_segment(num_sentence, original_sentence, args.n_split)
        segments = _split_sentence_indices(num_sentence, args.n_split, args.split_type)

        new_sentences = []
        source_labels = []

        # Alternate segments: Machine (L), Human (H), Machine (L), ... starting with Machine
        for seg_idx, (start, end) in enumerate(segments):
            seg_sentences = original_sentence[start:end]
            if len(seg_sentences) == 0:
                continue

            is_machine = (seg_idx % 2 == 0)
            if not is_machine:
                new_sentences.extend(seg_sentences)
                source_labels.extend(["H"] * len(seg_sentences))
                continue

            sub_original_texts = " ".join(seg_sentences)
            prompt = user_prompts[idx].strip()
            print(f"[Sample {idx}] Segment {seg_idx+1}/{len(segments)} (Machine) input: {sub_original_texts}")
            messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': f'{prompt}\n{sub_original_texts}'},
            ]
            kwargs["messages"] = messages
            response = client.chat.completions.create(**kwargs)
            output = response.choices[0].message.content
            output = _clean_llm_prefix(output)
            print(f">>> OpenAI response: {output}")

            # ---- 切分 LLM 返回的句子，并对齐句子数 ----
            _, rewritten_sentences = count_sentences(output)

            new_sentences.extend(rewritten_sentences)
            source_labels.extend(["L"] * len(rewritten_sentences))

        sampled_sentence_list.append(new_sentences)
        response_list.append(" ".join(new_sentences))
        source_label_list.append(source_labels)
    return response_list, sampled_sentence_list, source_label_list


def mistralai_sampler(original_texts, args):
    """Rewrite alternating segments using a local/open-source instruct model.

    Input/output format matches openai_sampler:
      returns (response_list, sampled_sentence_list, source_label_list)
    where source labels are "L" (machine) and "H" (human).
    """
    model_name = "mistralai/Ministral-8B-Instruct-2410"
    n_samples = len(original_texts)

    tokenizer = load_tokenizer(model_name, args.cache_dir)
    model = load_model(model_name, args.device, args.cache_dir)
    model.eval()

    prompt = "You are a rewriting expert and you would rewrite the text without missing the original details. Return ONLY the rewritten version. Do not explain changes, do not give multiple options, and do not add commentary. \n\n Original text: \"{}\" Here is the rewritten version: \n\n"

    def _generate_rewrite(segment_text: str) -> str:
        segment_text_ntokens = tokenizer(segment_text, return_tensors="pt", padding=True, return_token_type_ids=False)['input_ids'].shape[1]
        input_ids = tokenizer(prompt.format(segment_text), return_tensors="pt", padding=True, return_token_type_ids=False).to(args.device)
        prompt_lens = input_ids['input_ids'].shape[1]

        gen_kwargs = {
            "do_sample": True,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            'min_new_tokens': int(0.8*segment_text_ntokens),
            'max_new_tokens': int(1.2*segment_text_ntokens),
        }
        if getattr(args, "do_top_p", False):
            gen_kwargs["top_p"] = args.top_p
        if getattr(args, "do_top_k", False):
            gen_kwargs["top_k"] = args.top_k
        if getattr(args, "do_temperature", False):
            gen_kwargs["temperature"] = args.temperature

        output_ids = model.generate(**input_ids, **gen_kwargs)

        # Decode only the newly generated part
        new_tokens = output_ids[0, prompt_lens:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        return text.strip()

    sampled_sentence_list = []
    source_label_list = []
    response_list = []

    for idx in tqdm.tqdm(range(n_samples)):
        num_sentence, original_sentence = count_sentences(original_texts[idx])
        num_sentence, original_sentence = deduce_longer_segment(num_sentence, original_sentence, args.n_split)
        segments = _split_sentence_indices(num_sentence, args.n_split, args.split_type)

        new_sentences = []
        source_labels = []

        for seg_idx, (start, end) in enumerate(segments):
            seg_sentences = original_sentence[start:end]
            if len(seg_sentences) == 0:
                continue

            is_machine = (seg_idx % 2 == 0)
            if not is_machine:
                new_sentences.extend(seg_sentences)
                source_labels.extend(["H"] * len(seg_sentences))
                continue

            segment_text = " ".join(seg_sentences)
            print(f"[Sample {idx}] Segment {seg_idx+1}/{len(segments)} (Machine/Mistral) input: {segment_text}")
            output = _generate_rewrite(segment_text)
            print(f">>> Mistral response: {output}")

            _, rewritten_sentences = count_sentences(output)

            new_sentences.extend(rewritten_sentences)
            source_labels.extend(["L"] * len(rewritten_sentences))

        response_list.append(" ".join(new_sentences))
        sampled_sentence_list.append(new_sentences)
        source_label_list.append(source_labels)

    return response_list, sampled_sentence_list, source_label_list

def qwen_sampler(original_texts, args):
    """Rewrite alternating segments using a local/open-source instruct model.

    Input/output format matches openai_sampler:
      returns (response_list, sampled_sentence_list, source_label_list)
    where source labels are "L" (machine) and "H" (human).
    """
    model_name = "Qwen/Qwen3-4B"
    n_samples = len(original_texts)

    tokenizer = load_tokenizer(model_name, args.cache_dir)
    model = load_model(model_name, args.device, args.cache_dir)
    model.eval()

    prompt = "You are a English rewriting expert and you would rewrite the text without missing the original details. Return ONLY the rewritten version. Do not explain changes, do not give multiple options, and do not add commentary. \n\n Original text: \"{}\" Here is the rewritten version: \n\n"

    def _generate_rewrite(segment_text: str) -> str:
        segment_text_ntokens = tokenizer(segment_text, return_tensors="pt", padding=True, return_token_type_ids=False)['input_ids'].shape[1]
        prompt_segment_text = prompt.format(segment_text)
        prompt_segment_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_segment_text}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False # Turn of thinking modes
        )
        input_ids = tokenizer(prompt_segment_text, return_tensors="pt", padding=True, return_token_type_ids=False).to(args.device)
        prompt_lens = input_ids['input_ids'].shape[1]

        gen_kwargs = {
            "do_sample": True,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            'min_new_tokens': int(0.8*segment_text_ntokens),
            'max_new_tokens': int(1.2*segment_text_ntokens),
        }
        if getattr(args, "do_top_p", False):
            gen_kwargs["top_p"] = args.top_p
        if getattr(args, "do_top_k", False):
            gen_kwargs["top_k"] = args.top_k
        if getattr(args, "do_temperature", False):
            gen_kwargs["temperature"] = args.temperature

        output_ids = model.generate(**input_ids, **gen_kwargs)

        # Decode only the newly generated part
        new_tokens = output_ids[0, prompt_lens:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        return text.strip()

    sampled_sentence_list = []
    source_label_list = []
    response_list = []

    for idx in tqdm.tqdm(range(n_samples)):
        num_sentence, original_sentence = count_sentences(original_texts[idx])
        num_sentence, original_sentence = deduce_longer_segment(num_sentence, original_sentence, args.n_split)
        segments = _split_sentence_indices(num_sentence, args.n_split, args.split_type)

        new_sentences = []
        source_labels = []

        for seg_idx, (start, end) in enumerate(segments):
            seg_sentences = original_sentence[start:end]
            if len(seg_sentences) == 0:
                continue

            is_machine = (seg_idx % 2 == 0)
            if not is_machine:
                new_sentences.extend(seg_sentences)
                source_labels.extend(["H"] * len(seg_sentences))
                continue

            segment_text = " ".join(seg_sentences)
            print(f"[Sample {idx}] Segment {seg_idx+1}/{len(segments)} (Machine/Qwen) input: {segment_text}")
            output = _generate_rewrite(segment_text)
            print(f">>> Qwen response: {output}")

            _, rewritten_sentences = count_sentences(output)

            new_sentences.extend(rewritten_sentences)
            source_labels.extend(["L"] * len(rewritten_sentences))

        response_list.append(" ".join(new_sentences))
        sampled_sentence_list.append(new_sentences)
        source_label_list.append(source_labels)

    return response_list, sampled_sentence_list, source_label_list

def gemma2_sampler(original_texts, args):
    """
    Rewrite alternating segments using google/gemma-2-2b-it.

    Input/output format matches openai_sampler / qwen_sampler:
      returns (response_list, sampled_sentence_list, source_label_list)
    where source labels are "L" (machine) and "H" (human).

    Requirements:
      - count_sentences(text) -> (num_sentence, sentence_list)
      - deduce_longer_segment(num_sentence, sentence_list, args.n_split) -> (num_sentence, sentence_list)
      - _split_sentence_indices(num_sentence, args.n_split, args.split_type) -> list[(start,end)] with python slicing indices
    """
    model_name = "google/gemma-2-2b-it"
    n_samples = len(original_texts)

    cache_dir = getattr(args, "cache_dir", None)
    device = getattr(args, "device", "cuda" if torch.cuda.is_available() else "cpu")

    # Gemma-2 官方示例：AutoTokenizer + AutoModelForCausalLM，并建议 apply_chat_template 走聊天模板。:contentReference[oaicite:2]{index=2}
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        torch_dtype=(torch.bfloat16 if (device.startswith("cuda") and torch.cuda.is_bf16_supported()) else torch.float16)
        if device.startswith("cuda") else None,
        device_map="auto" if device.startswith("cuda") else None,
    )

    # 如果没用 device_map="auto"（如 cpu），手动搬运
    if not device.startswith("cuda"):
        model = model.to(device)

    model.eval()

    # Gemma 有时 tokenizer.pad_token_id 可能为空；用 eos 兜底更稳
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    prompt = (
        "You are an English rewriting expert and you would rewrite the text without missing the original details. "
        "Return ONLY the rewritten version. Do not explain changes, do not give multiple options, and do not add commentary.\n\n"
        'Original text: "{}"\n\n'
        "Here is the rewritten version:\n"
    )

    def _count_tokens_plain(text: str) -> int:
        ids = tokenizer(text, add_special_tokens=False).get("input_ids", [])
        return max(1, len(ids))

    @torch.inference_mode()
    def _generate_rewrite(segment_text: str) -> str:
        segment_text_ntokens = _count_tokens_plain(segment_text)
        user_prompt = prompt.format(segment_text)

        # 官方模型卡示例：messages + tokenizer.apply_chat_template(...)。:contentReference[oaicite:3]{index=3}
        messages = [{"role": "user", "content": user_prompt}]
        inputs = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True,
        )

        # inputs 搬到模型所在设备（device_map="auto" 时用 model.device 不一定准确，稳妥用 inputs 的 device 与 model 的第一参数设备一致）
        target_device = next(model.parameters()).device
        inputs = {k: v.to(target_device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]

        gen_kwargs = {
            "do_sample": True,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "min_new_tokens": int(0.8 * segment_text_ntokens),
            "max_new_tokens": int(1.2 * segment_text_ntokens),
        }
        if getattr(args, "do_top_p", False):
            gen_kwargs["top_p"] = args.top_p
        if getattr(args, "do_top_k", False):
            gen_kwargs["top_k"] = args.top_k
        if getattr(args, "do_temperature", False):
            gen_kwargs["temperature"] = args.temperature

        output_ids = model.generate(**inputs, **gen_kwargs)

        # Decode only newly generated tokens
        new_tokens = output_ids[0, prompt_len:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        return text.strip()

    sampled_sentence_list = []
    source_label_list = []
    response_list = []

    for idx in tqdm.tqdm(range(n_samples)):
        num_sentence, original_sentence = count_sentences(original_texts[idx])
        num_sentence, original_sentence = deduce_longer_segment(num_sentence, original_sentence, args.n_split)
        segments = _split_sentence_indices(num_sentence, args.n_split, args.split_type)

        new_sentences = []
        source_labels = []

        for seg_idx, (start, end) in enumerate(segments):
            seg_sentences = original_sentence[start:end]
            if len(seg_sentences) == 0:
                continue

            is_machine = (seg_idx % 2 == 0)
            if not is_machine:
                new_sentences.extend(seg_sentences)
                source_labels.extend(["H"] * len(seg_sentences))
                continue

            segment_text = " ".join(seg_sentences)
            print(f"[Sample {idx}] Segment {seg_idx+1}/{len(segments)} (Machine/Gemma2) input: {segment_text}")
            output = _generate_rewrite(segment_text)
            print(f">>> Gemma2 response: {output}")

            _, rewritten_sentences = count_sentences(output)

            new_sentences.extend(rewritten_sentences)
            source_labels.extend(["L"] * len(rewritten_sentences))

        response_list.append(" ".join(new_sentences))
        sampled_sentence_list.append(new_sentences)
        source_label_list.append(source_labels)

    return response_list, sampled_sentence_list, source_label_list


def gemma_sampler(original_texts, args):
    """
    Rewrite alternating segments using google/gemma-3-4b-it.

    Input/output format matches qwen_sampler:
      returns (response_list, sampled_sentence_list, source_label_list)
    where source labels are "L" (machine) and "H" (human).
    """
    model_name = "google/gemma-3-4b-it"
    n_samples = len(original_texts)

    # Gemma 3 官方推荐用 AutoProcessor + Gemma3ForConditionalGeneration
    # 参考 HF 模型页示例：processor.apply_chat_template(...).to(model.device, dtype=...)
    processor = AutoProcessor.from_pretrained(model_name, cache_dir=getattr(args, "cache_dir", None))
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_name,
        cache_dir=getattr(args, "cache_dir", None),
        torch_dtype=getattr(args, "torch_dtype", None)  # 可选：你也可以在 args 里塞 torch.bfloat16 / torch.float16
    )

    device = getattr(args, "device", "cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # processor 内部带 tokenizer
    tok = getattr(processor, "tokenizer", None)
    if tok is not None:
        # Gemma 系列经常需要 pad_token_id；缺省时用 eos 兜底
        if tok.pad_token_id is None and tok.eos_token_id is not None:
            tok.pad_token_id = tok.eos_token_id
        # 生成任务通常 left padding 更稳（尤其 batch 时）；单条也无害
        tok.padding_side = "left"

    prompt_tpl = (
        "You are an English rewriting expert. You will rewrite the text without missing any original details. "
        "Return ONLY the rewritten version. Do not explain changes, do not give multiple options, and do not add commentary.\n\n"
        "Original text:\n\"{}\"\n\nRewritten version:\n"
    )

    def _segment_ntokens(segment_text: str) -> int:
        # 用 processor.tokenizer 估算输入 token 数；只用于设定 min/max_new_tokens
        if tok is None:
            # 兜底：不用 token 数控制长度
            return max(1, len(segment_text.split()))
        ids = tok(segment_text, add_special_tokens=False).get("input_ids", [])
        return max(1, len(ids))

    @torch.inference_mode()
    def _generate_rewrite(segment_text: str) -> str:
        seg_ntok = _segment_ntokens(segment_text)
        user_text = prompt_tpl.format(segment_text)

        # Gemma 3 的 chat template（文本-only 也按 IT 模式走）
        # 注意：Gemma 3 示例里 messages 的 content 是 [{"type":"text","text":...}]
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": user_text}],
            },
        ]

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        # dtype：优先用 bfloat16（若你环境支持），否则保持默认
        dtype = getattr(args, "torch_dtype", None)
        if dtype is None:
            # 经验：cuda 上默认用 bf16/fp16 更省显存；不强制，按需开启
            if "cuda" in str(device):
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        inputs = {k: v.to(device=device, dtype=(dtype if v.is_floating_point() else None)) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[-1]

        gen_kwargs = {
            "do_sample": True,
            "max_new_tokens": int(1.2 * seg_ntok),
            "min_new_tokens": int(0.8 * seg_ntok),
        }

        # 采样参数（沿用你 qwen 的 args 逻辑）
        if getattr(args, "do_top_p", False):
            gen_kwargs["top_p"] = args.top_p
        if getattr(args, "do_top_k", False):
            gen_kwargs["top_k"] = args.top_k
        if getattr(args, "do_temperature", False):
            gen_kwargs["temperature"] = args.temperature

        # pad/eos（尽量给齐）
        if tok is not None:
            if tok.eos_token_id is not None:
                gen_kwargs["eos_token_id"] = tok.eos_token_id
            if tok.pad_token_id is not None:
                gen_kwargs["pad_token_id"] = tok.pad_token_id

        out = model.generate(**inputs, **gen_kwargs)

        # 只解码新生成部分
        new_tokens = out[0][input_len:]
        text = processor.decode(new_tokens, skip_special_tokens=True)
        return text.strip()

    sampled_sentence_list = []
    source_label_list = []
    response_list = []

    for idx in tqdm.tqdm(range(n_samples)):
        num_sentence, original_sentence = count_sentences(original_texts[idx])
        num_sentence, original_sentence = deduce_longer_segment(num_sentence, original_sentence, args.n_split)
        segments = _split_sentence_indices(num_sentence, args.n_split, args.split_type)

        new_sentences = []
        source_labels = []

        for seg_idx, (start, end) in enumerate(segments):
            seg_sentences = original_sentence[start:end]
            if len(seg_sentences) == 0:
                continue

            is_machine = (seg_idx % 2 == 0)
            if not is_machine:
                new_sentences.extend(seg_sentences)
                source_labels.extend(["H"] * len(seg_sentences))
                continue

            segment_text = " ".join(seg_sentences)
            print(f"[Sample {idx}] Segment {seg_idx+1}/{len(segments)} (Machine/Gemma) input: {segment_text}")
            output = _generate_rewrite(segment_text)
            print(f">>> Gemma response: {output}")

            _, rewritten_sentences = count_sentences(output)

            new_sentences.extend(rewritten_sentences)
            source_labels.extend(["L"] * len(rewritten_sentences))

        response_list.append(" ".join(new_sentences))
        sampled_sentence_list.append(new_sentences)
        source_label_list.append(source_labels)

    return response_list, sampled_sentence_list, source_label_list


def mistralai_7b_sampler(original_texts, args):
    """
    Rewrite alternating segments using mistralai/Mistral-7B-Instruct-v0.3.

    Input/output format matches qwen_sampler:
      returns (response_list, sampled_sentence_list, source_label_list)
    where source labels are "L" (machine) and "H" (human).

    Requires your existing helpers:
      - count_sentences(text) -> (num_sentence, sentence_list)
      - deduce_longer_segment(num_sentence, sentence_list, args.n_split) -> (num_sentence, sentence_list)
      - _split_sentence_indices(num_sentence, args.n_split, args.split_type) -> list[(start,end)] (python slice indices)
    """
    model_fullname = "mistralai/Mistral-7B-Instruct-v0.3"
    n_samples = len(original_texts)

    cache_dir = getattr(args, "cache_dir", None)
    device = getattr(args, "device", "cuda" if torch.cuda.is_available() else "cpu")

    # Load
    tokenizer = AutoTokenizer.from_pretrained(model_fullname, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_fullname,
        cache_dir=cache_dir,
        torch_dtype=torch.float16 if device.startswith("cuda") else None,
        device_map="auto" if device.startswith("cuda") else None,
    )

    if not device.startswith("cuda"):
        model = model.to(device)

    model.eval()

    # Safer padding defaults
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    prompt = (
        "You are an English rewriting expert and you would rewrite the text without missing the original details. "
        "Return ONLY the rewritten version. Do not explain changes, do not give multiple options, and do not add commentary.\n\n"
        'Original text: "{}"\n\n'
        "Here is the rewritten version:\n"
    )

    def _count_tokens_plain(text: str) -> int:
        ids = tokenizer(text, add_special_tokens=False).get("input_ids", [])
        return max(1, len(ids))

    def _build_chat_prompt(user_text: str) -> str:
        """
        Prefer tokenizer's chat template if available; otherwise fall back to Mistral [INST] format.
        """
        messages = [{"role": "user", "content": user_text}]
        if hasattr(tokenizer, "apply_chat_template"):
            # tokenize=False so we can measure prompt_len after tokenization
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        # Fallback (works for classic Mistral-Instruct style)
        return f"<s>[INST] {user_text.strip()} [/INST]"

    @torch.inference_mode()
    def _generate_rewrite(segment_text: str) -> str:
        seg_ntok = _count_tokens_plain(segment_text)
        user_prompt = prompt.format(segment_text)

        chat_text = _build_chat_prompt(user_prompt)
        inputs = tokenizer(
            chat_text,
            return_tensors="pt",
            padding=True,
            return_token_type_ids=False,
        )

        # Move to the model's device
        target_device = next(model.parameters()).device
        inputs = {k: v.to(target_device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]

        gen_kwargs = {
            "do_sample": True,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "min_new_tokens": int(0.8 * seg_ntok),
            "max_new_tokens": int(1.2 * seg_ntok),
        }
        if getattr(args, "do_top_p", False):
            gen_kwargs["top_p"] = args.top_p
        if getattr(args, "do_top_k", False):
            gen_kwargs["top_k"] = args.top_k
        if getattr(args, "do_temperature", False):
            gen_kwargs["temperature"] = args.temperature

        output_ids = model.generate(**inputs, **gen_kwargs)

        # Decode only the newly generated part
        new_tokens = output_ids[0, prompt_len:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        return text.strip()

    sampled_sentence_list = []
    source_label_list = []
    response_list = []

    for idx in tqdm.tqdm(range(n_samples)):
        num_sentence, original_sentence = count_sentences(original_texts[idx])
        num_sentence, original_sentence = deduce_longer_segment(num_sentence, original_sentence, args.n_split)
        segments = _split_sentence_indices(num_sentence, args.n_split, args.split_type)

        new_sentences = []
        source_labels = []

        for seg_idx, (start, end) in enumerate(segments):
            seg_sentences = original_sentence[start:end]
            if len(seg_sentences) == 0:
                continue

            is_machine = (seg_idx % 2 == 0)
            if not is_machine:
                new_sentences.extend(seg_sentences)
                source_labels.extend(["H"] * len(seg_sentences))
                continue

            segment_text = " ".join(seg_sentences)
            print(f"[Sample {idx}] Segment {seg_idx+1}/{len(segments)} (Machine/Mistral7B) input: {segment_text}")
            output = _generate_rewrite(segment_text)
            print(f">>> Mistral response: {output}")

            _, rewritten_sentences = count_sentences(output)
            new_sentences.extend(rewritten_sentences)
            source_labels.extend(["L"] * len(rewritten_sentences))

        response_list.append(" ".join(new_sentences))
        sampled_sentence_list.append(new_sentences)
        source_label_list.append(source_labels)

    return response_list, sampled_sentence_list, source_label_list


def claude_sample(original_texts, task, args) -> str:
    def _clean_claude_generated_text(text: str) -> str:
        """
        移除类似 "Here's xxx:" 这种前缀提示语，只保留正文
        """
        # 匹配 "Here's ..." 或 "Here is ..." 后跟冒号的部分
        cleaned = re.sub(r"(?i)\bhere(?:'s| is)\s+[^:：]+[:：]\s*", "", text)
        return cleaned.strip()

    from anthropic import Anthropic
    client = Anthropic()
    model_full_name_list = {'claude-3-5-haiku': "claude-3-5-haiku-20241022"}
    model_full_name = model_full_name_list[args.model_name]
    n_samples = len(original_texts)

    if task == "rewrite":
        system_prompt = 'You are a professional rewriting expert and you can help paraphrasing this paragraph in English without missing the original details. Please keep the length of the rewritten text similar to the original text. Return ONLY the rewritten version. Do not explain changes, do not give multiple options, and do not add commentary.'
        user_prompts = ['Please rewrite:'] * n_samples
    elif task == "polish":
        system_prompt = 'You are a professional polishing expert and you can help polishing this paragraph. Return ONLY the polished version. Do not explain changes, do not give multiple options, and do not add commentary.'
        with open("./data/polish_prompt.json","r") as p:
            user_prompts = json.load(p)['out_prompt']
    elif task == "expand":
        system_prompt = 'You are a professional writing expert and you can help expanding this paragraph. Return ONLY the expanded version. Do not explain, do not give multiple options, and do not add commentary.'
        with open("./data/expand_prompt.json","r") as p:
            user_prompts = json.load(p)['prompt']
    req = {
        "system": system_prompt,
        "temperature": args.temperature if args.do_temperature else None,
        "top_p": args.top_p if args.do_top_p else None,
        "top_k": args.top_k if args.do_top_k else None,
    }

    retries = 10
    response_list =[]
    for idx in tqdm.tqdm(range(n_samples)):
        original_text = original_texts[idx]
        print(f"Original text: {original_text}")
        prompt = user_prompts[idx].strip()
        for i in range(retries):
            try:
                response = client.messages.create(
                    model=model_full_name, 
                    max_tokens=1000,
                    messages=[{"role": "user", "content": f'{prompt} {original_text}'}],
                    **{k: v for k, v in req.items() if v is not None}
                )
                continue
            except Exception as e:
                wait_time = (2 ** i) + random.uniform(0, 1)
                print(f"Request failed ({e}), retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)
        response = response.content[0].text.strip()
        output = _clean_claude_generated_text(response)
        print(f">>> Claude response: {output}")
        response_list.append(output)

    return response_list

def gemini_sample(original_texts, task, args) -> str:
    from google import genai
    from google.genai import types
    client = genai.Client()
    n_samples = len(original_texts)

    if task == "rewrite":
        system_prompt = 'You are a professional rewriting expert and you can help paraphrasing this paragraph in English without missing the original details. Please keep the length of the rewritten text similar to the original text. Return ONLY the rewritten version. Do not explain changes, do not give multiple options, and do not add commentary.'
        user_prompts = ['Please rewrite:'] * len(original_texts)
    elif task == "polish":
        system_prompt = 'You are a professional polishing expert and you can help polishing this paragraph. Return ONLY the polished version. Do not explain changes, do not give multiple options, and do not add commentary.'
        with open("./data/polish_prompt.json","r") as p:
            user_prompts = json.load(p)['out_prompt']
    elif task == "expand":
        system_prompt = 'You are a professional writing expert and you can help expanding this paragraph. Return ONLY the expanded version. Do not explain, do not give multiple options, and do not add commentary.'
        with open("./data/expand_prompt.json","r") as p:
            user_prompts = json.load(p)['prompt']

    max_retries = 5
    base_delay = 2
    response_list =[]
    for idx in tqdm.tqdm(range(n_samples)):
        prompt = user_prompts[idx].strip()
        original_text = original_texts[idx]
        print(f"Original text: {original_text}")
        params = {"model": args.model_name, "contents": f'{prompt}\n{original_text}',}
        response = None
        for i in range(max_retries):
            try:
                response = client.models.generate_content(
                    **params,
                    config=types.GenerateContentConfig(
                        top_p=args.top_p if args.do_top_p else None,
                        top_k=args.top_k if args.do_top_k else None,
                        temperature=args.temperature if args.do_temperature else None,
                        seed=args.seed,
                        candidate_count=1,
                        system_instruction=system_prompt,
                    ),
                )
                break
            except Exception as e:
                print(f"Error: {e}, retry {i+1}/{max_retries}")
                time.sleep(base_delay * (2 ** i))  # exponential backoff
        if response is None:
            raise RuntimeError(f"Failed after {max_retries} retries for sample {idx}")
        
        output = response.text.strip()
        print(f">>> Gemini response: {output}")
        response_list.append(output)

    return response_list

def save_data(output_file, args, data):
    # write args to file
    args_file = f"{output_file}.args.json"
    with open(args_file, "w") as fout:
        json.dump(args.__dict__, fout, indent=4)
        print(f"Args written into {args_file}")

    # write the data to a json file in the save folder
    data_file = f"{output_file}.raw_data.json"
    with open(data_file, "w") as fout:
        json.dump(data, fout, indent=4)
        print(f"Raw data written into {data_file}")

def forward(args):
    if args.dataset == 'yelp':
        args.dataset = 'yelp_polarity'
    if args.dataset == "bbc":
        args.dataset = 'gopalkalpande/bbc-news-summary'
    print(f'Loading dataset {args.dataset}...')
    dataset_keys = {'xsum': 'document', 'squad': 'context', 'writing': 'document', 'essay': 'document', 'yelp_polarity': 'text', 'gopalkalpande/bbc-news-summary': 'Summaries'}

    original_texts = load_data(args, args.dataset, dataset_keys[args.dataset] if args.dataset in dataset_keys else None)

    # tokenizer = load_tokenizer('gpt-neo-2.7B', cache_dir=args.cache_dir)
    # # keep only examples with <= 2048 tokens according to base_tokenizer
    # # this step has the extra effect of removing examples with low-quality/garbage content
    # tokenized_data = tokenizer(original_texts)
    # original_texts = [x for x, y in zip(original_texts, tokenized_data["input_ids"]) if len(y) <= 2048]

    # print stats about remaining data
    print(f"Total number of samples: {len(original_texts)}")
    print(f"Average number of words: {np.mean([len(x.split()) for x in original_texts])}")

    original_texts = original_texts[:min(args.n_samples, len(original_texts))]

    if args.model_name in ['gpt-3.5-turbo', 'gpt-4', 'gpt-4o']:
        sampled_texts, sampled_sentence_list, source_label_list = openai_sampler(original_texts, args)
    elif args.model_name in ['mistralai-8b-instruct']:
        sampled_texts, sampled_sentence_list, source_label_list = mistralai_sampler(original_texts, args)
    elif args.model_name in ['mistralai-7b-instruct']:
        sampled_texts, sampled_sentence_list, source_label_list = mistralai_7b_sampler(original_texts, args)
    elif args.model_name in ['qwen-4b-instruct']:
        sampled_texts, sampled_sentence_list, source_label_list = qwen_sampler(original_texts, args)
    elif args.model_name in ['gemma-1b-instruct', 'gemma-4b-instruct']:
        sampled_texts, sampled_sentence_list, source_label_list = gemma_sampler(original_texts, args)
    elif args.model_name in ['gemma-2b-instruct']:
        sampled_texts, sampled_sentence_list, source_label_list = gemma2_sampler(original_texts, args)
    elif args.model_name == 'gemini-2.5-flash':
        sampled_texts = gemini_sample(original_texts, args.task, args)
    elif args.model_name == 'claude-3-5-haiku':
        sampled_texts = claude_sample(original_texts, args.task, args)

    if args.dataset == 'yelp_polarity':
        args.dataset = 'yelp'
    if args.dataset == "gopalkalpande/bbc-news-summary":
        args.dataset = 'bbc'

    data = {"original": [], "sampled": [], "sampled_sentence": [], "source_label": []}
    for o, s, s_sentence, label in zip(original_texts, sampled_texts, sampled_sentence_list, source_label_list):
        # add to the data
        data["original"].append(o)
        data["sampled"].append(s)
        data["sampled_sentence"].append(s_sentence)
        data["source_label"].append(label)

    save_data(args.output_file, args, data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', type=str, default="./exp_location_single_cp/data/squad_gpt-4o_rewrite")
    parser.add_argument('--task', type=str, default="rewrite", choices=["rewrite", "polish", "expand", "generation", "summary"])
    parser.add_argument('--dataset', type=str, default="squad", choices=['xsum', 'squad', 'writing', 'pubmed', 'essay', 'yelp', 'bbc'])
    parser.add_argument('--n_samples', type=int, default=25)
    parser.add_argument(
        '--base_model_name',
        type=str,
        default="gpt-4o",
        choices=["gpt-4o", "gemini-2.5-flash", "claude-3-5-haiku", "qwen-4b-instruct", "mistralai-7b-instruct", "mistralai-8b-instruct", "gemma-4b-instruct", "gemma-2b-instruct"],
    )
    parser.add_argument('--n_split', type=int, default=2, help='Number of segments K (>=2). Segments alternate Machine/Human starting with Machine.')
    parser.add_argument('--split_type', type=str, default='equal_len', choices=['random', 'equal_len'])
    parser.add_argument('--max_new_tokens', type=int, default=1000)
    parser.add_argument('--do_top_k', action='store_true')
    parser.add_argument('--top_k', type=int, default=40)
    parser.add_argument('--do_top_p', action='store_true')
    parser.add_argument('--top_p', type=float, default=0.96)
    parser.add_argument('--do_temperature', action='store_true')
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--cache_dir', type=str, default="../cache")
    args = parser.parse_args()
    args.model_name = args.base_model_name

    if args.n_split < 2:
        raise ValueError(f"--n_split must be >= 2, got {args.n_split}")

    set_seed(args.seed)

    forward(args)
    
    