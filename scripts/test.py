from peft import AutoPeftModelForCausalLM
adapter_dir = "/root/project/Practice/code/scripts/FineTune/ckpt/scoring_model"
model = AutoPeftModelForCausalLM.from_pretrained(adapter_dir, local_files_only=True)
 