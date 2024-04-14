from transformers import AutoTokenizer


for size in ['125m', '1.3b', '2.7b', '6.7b', '13b']:
    tokenizer = AutoTokenizer.from_pretrained(f'facebook/opt-{size}', use_fast=False, cache_dir="tokenizers")