from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import argparse

#Download Models and Tokenizers

argparser = argparse.ArgumentParser()
argparser.add_argument("--local_cache", default=False, action="store_true")
argparser.add_argument('--model', type=str)

args = argparser.parse_args()

hf_token = "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"

lmharness = False
data = False

if args.model == "llama2":
    sizes = ['7b', '13b'] #, '70b']
    model_name_holder = 'meta-llama/Llama-2-{}-hf'
elif args.model == 'llama3_1':
    sizes = ['8B']#, '70B', '405B']
    model_name_holder = "mistralai/Mistral-{}-v0.3"
elif args.model == "mistral":
    sizes = ['7B']
    model_name_holder = "mistralai/Mistral-{size}-v0.3"
elif args.model == "opt":
    sizes = ['125m', '350m', '1.3b', '2.7b', '6.7b', '13b', '30b', '66b']
    model_name_holder = 'facebook/opt-{}'
elif args.model == "lmharness":
    lmharness = True
elif args.model == "data":
    data = True
else:
    raise NotImplementedError


if lmharness:
    # Download LM Harness Data
    import lm_eval

    results = lm_eval.simple_evaluate(
        model="hf",
        model_args=f"pretrained=facebook/opt-125m,dtype=half,parallelize=True,cache_dir=llm_weights",
        tasks=["mmlu", "piqa", "arc_easy", "arc_challenge", "winogrande", "openbookqa"],
        verbosity="ERROR"
    )


if data:
    #Download Pruning, Evaluation, and Fine-tuning Data
    from datasets import load_dataset
    cache_dir = "data"
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train', cache_dir=cache_dir)
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test', cache_dir=cache_dir)

    traindata.save_to_disk(f"{cache_dir}/wikitext-train.pt")
    testdata.save_to_disk(f"{cache_dir}/wikitext-test.pt")

    try:
        traindata = load_dataset('allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train', cache_dir=cache_dir)
        valdata = load_dataset('allenai/c4', 'allenai--c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation', cache_dir=cache_dir)
    except:
        traindata = load_dataset('allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train', cache_dir=cache_dir)
        valdata = load_dataset('allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation', cache_dir=cache_dir)

    traindata.save_to_disk(f"{cache_dir}/c4-train.pt")
    valdata.save_to_disk(f"{cache_dir}/c4-val.pt")

    try:
        raw_datasets = load_dataset('allenai/c4',
                                    'allenai--c4',
                                    data_files={'train': 'en/c4-train.00000-of-01024.json.gz',
                                                'validation': 'en/c4-validation.00000-of-00008.json.gz'},
                                    cache_dir=cache_dir)
    except:
        raw_datasets = load_dataset('allenai/c4',
                                    data_files={'train': 'en/c4-train.00000-of-01024.json.gz',
                                                'validation': 'en/c4-validation.00000-of-00008.json.gz'},
                                    cache_dir=cache_dir)

    raw_datasets.save_to_disk(f"{cache_dir}/c4-raw.pt")
    exit()



if args.local_cache:
    model_cache_dir = "llm_weights"
    tokenizer_cache_dir = "tokenizers"
else:
    model_cache_dir = None
    tokenizer_cache_dir = None


for size in sizes:
    model_name = model_name_holder.format(size)
    print("Loading Model: ", model_name)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            cache_dir=model_cache_dir,
            low_cpu_mem_usage=True,
            token=hf_token,
            # device_map='auto'
            )
    except:
        pass
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, token=hf_token, cache_dir=tokenizer_cache_dir)


