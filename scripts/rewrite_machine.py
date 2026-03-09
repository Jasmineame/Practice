import torch
torch._dynamo.config.disable = True
import custom_datasets
from model import load_tokenizer, load_model
import re

# PROMPT = "You are a rewriting expert and you would rewrite the text without missing the original details. Return ONLY the rewritten version. Do not explain changes, do not give multiple options, and do not add commentary. \n\n Original text: \"{}\" Here is the rewritten version: \n\n"
PROMPT = "You are a rewriting expert and you would rewrite the text without missing the original details. Return ONLY the rewritten version. Do not explain changes, do not give multiple options, and do not add commentary. \n\n Original text: \"{}\""

class PrefixSampler:
    def __init__(self, args):
        self.args = args
        self.base_tokenizer = load_tokenizer(args.rewrite_model_name, args.cache_dir)
        self.base_model = load_model(args.rewrite_model_name, args.device, args.cache_dir, local_files_only=args.local_files_only)
        self.base_model.eval()
        # self.pipe = pipeline("text-generation", model=self.base_model, tokenizer=self.base_tokenizer, device=torch.cuda.current_device())

        # try:
        #     self.base_model = BetterTransformer.transform(self.base_model)
        #     print("Successfully transformed model with BetterTransformer.")
        # except Exception as e:
        #     self.base_model = torch.compile(self.base_model, mode="reduce-overhead")
        #     print(f"[Warning] BetterTransformer transformation failed. Falling back to using torch.compile. Error message: {e}")
        self.base_model = torch.compile(self.base_model, mode="reduce-overhead")

    def _sample_rewrite_text_from_model(self, texts, rewrite_model_name=None):
        texts_num_tokens = self.base_tokenizer(texts, return_tensors="pt", padding=True, return_token_type_ids=False)['input_ids'].shape[1]
        prompt_texts = [PROMPT.format(o) for o in texts] 

        sampling_kwargs = {'temperature': self.args.temperature}
        if self.args.do_top_p:
            sampling_kwargs['top_p'] = self.args.top_p
        elif self.args.do_top_k:
            sampling_kwargs['top_k'] = self.args.top_k

        sampling_kwargs['min_new_tokens'] = int(0.8*texts_num_tokens)
        sampling_kwargs['max_new_tokens'] = int(2*texts_num_tokens)
        sampling_kwargs["eos_token_id"] = self.base_tokenizer.eos_token_id
        sampling_kwargs['pad_token_id'] = self.base_tokenizer.eos_token_id

        if "Qwen3.5-4B" in rewrite_model_name:
            decoded = ['' for _ in range(len(texts))]
            for idx, prompt_text in enumerate(prompt_texts):
                message = [
                {"role": "user", "content": prompt_text}
                ]
                text = self.base_tokenizer.apply_chat_template(
                    message,
                    tokenize=False
                )
                model_inputs = self.base_tokenizer([text], return_tensors="pt").to(self.base_model.device)
                with torch.no_grad():
                    generated_ids = self.base_model.generate(
                                **model_inputs,
                                do_sample=True,
                                enable_thinking=False, # Switches between thinking and non-thinking modes. Default is True.
                                **sampling_kwargs
                            )
                output_ids = generated_ids[0][model_inputs.input_ids.shape[1]:].tolist()

                content = self.base_tokenizer.decode(output_ids, skip_special_tokens=True).strip()
                content = re.sub(r'assistant\n<think>\n\n</think>\n\n', '', content, flags=re.IGNORECASE)
                content = content.lstrip()
                decoded[idx] = content
            return decoded
        elif "Qwen3-4B" in rewrite_model_name:
            decoded = ['' for _ in range(len(texts))]
            for idx, prompt_text in enumerate(prompt_texts):
                message = [
                {"role": "user", "content": prompt_text}
                ]
                text = self.base_tokenizer.apply_chat_template(
                    message,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
                )
                model_inputs = self.base_tokenizer([text], return_tensors="pt").to(self.base_model.device)
                generated_ids = self.base_model.generate(
                            **model_inputs,
                            do_sample=True, 
                            enable_thinking=False, # Switches between thinking and non-thinking modes. Default is True.
                            **sampling_kwargs
                        )
                output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
                # parsing thinking content
                try:
                    # rindex finding 151668 (</think>)
                    index = len(output_ids) - output_ids[::-1].index(151668)
                except ValueError:
                    index = 0

                thinking_content = self.base_tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
                content = self.base_tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
                decoded[idx] = content
            return decoded
        else:
            decoded = ['' for _ in range(len(texts))]
            all_encoded = self.base_tokenizer(prompt_texts, return_tensors="pt", padding=True, return_token_type_ids=False).to(self.args.device)
            prompt_lens = all_encoded['input_ids'].shape[1]
            outputs = self.base_model.generate(**all_encoded, do_sample=True, **sampling_kwargs)
            gen_ids = outputs[:, prompt_lens:]
            decoded = self.base_tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            return decoded

    # def generate_samples(self, raw_data, batch_size):
    #     data = {
    #         "original": [],
    #         "sampled": [],
    #     }

    #     assert len(raw_data) % batch_size == 0
    #     for batch in range(len(raw_data) // batch_size):
    #         print('Generating samples for batch', batch, 'of', len(raw_data) // batch_size)
    #         original_text = raw_data[batch * batch_size:(batch + 1) * batch_size]
    #         sampled_text = self._sample_rewrite_text_from_model(original_text)

    #         for o, s in zip(original_text, sampled_text):
    #             # add to the data
    #             data["original"].append(o)
    #             data["sampled"].append(s)

    #     return data

    def generate_samples(self, raw_data, batch_size, rewrite_model_name=None):
        data = {
            "original": [],
            "sampled": [],
        }

        n = len(raw_data)
        if n == 0:
            return data
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        num_batches = (n + batch_size - 1) // batch_size  # ceil division

        for batch in range(num_batches):
            print("Generating samples for batch", batch, "of", num_batches)

            start = batch * batch_size
            end = min(start + batch_size, n)

            original_text = raw_data[start:end]
            sampled_text = self._sample_rewrite_text_from_model(original_text, rewrite_model_name=rewrite_model_name)

            # Safety check: model should return one sample per input
            if len(sampled_text) != len(original_text):
                raise RuntimeError(
                    f"Expected {len(original_text)} samples, got {len(sampled_text)} "
                    f"for batch {batch} (start={start}, end={end})."
                )

            for o, s in zip(original_text, sampled_text):
                data["original"].append(o)
                data["sampled"].append(s)

        return data
