from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


hf_token = "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"

llama2_sizes = ['7b', '13b', '70b']
opt_sizes = ['125m', '1.3b', '2.7b', '6.7b', '13b', '30b', '66b']

for size in llama2_sizes:
    # model_name = f'facebook/opt-{size}'
    model_name = f'meta-llama/Llama-2-{size}-hf'
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        cache_dir="llm_weights",
        low_cpu_mem_usage=True,
        token=hf_token,
        # device_map='auto'
        )
    print(model)
    exit()
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, cache_dir="tokenizers", token=hf_token)