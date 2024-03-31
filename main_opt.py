import argparse
import os 
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from importlib.metadata import version

from lib.prune_opt import prune_wanda, prune_magnitude, prune_sparsegpt, prune_ablate, check_sparsity, find_layers
from lib.eval import eval_ppl, eval_zero_shot
from lib.utils import get_layers_list

print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())


def conv1d_to_linear(conv1d):
    input_dim = conv1d.weight.shape[0]
    output_dim = conv1d.weight.shape[1]
    bias = conv1d.bias is not None
    layer = torch.nn.Linear(input_dim, output_dim, bias=bias)
    layer.weight.data = conv1d.weight.data.T
    if bias:
        layer.bias.data = conv1d.bias.data
    return layer


def convert_flashattn_checkpoint_to_hf(checkpoint):
    if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
            for key in list(checkpoint.keys()):
                new_key = key.replace('model.', '')
                if "embeddings.word_embeddings.weight" in new_key:
                    new_key = "transformer.wte.weight"
                if "embeddings.position_embeddings.weight" in new_key:
                    new_key = "transformer.wpe.weight"
                if "layers" in new_key:
                    new_key = new_key.replace("layers", "h")
                if "mixer" in new_key:
                    new_key = new_key.replace("mixer", "attn")
                if "Wqkv" in new_key:
                    new_key = new_key.replace("Wqkv", "c_attn")
                if "out_proj" in new_key:
                    new_key = new_key.replace("out_proj", "c_proj")
                if "norm" in new_key:
                    new_key = new_key.replace("norm", "ln_")
                if "fc1" in new_key:
                    new_key = new_key.replace("fc1",  "c_fc")
                if "fc2" in new_key:
                    new_key = new_key.replace("fc2", "c_proj")
                if "wte" in new_key or "wpe" in new_key or "weight" not in new_key or "lora" in new_key or "lm_head" in new_key:
                    checkpoint[new_key] = checkpoint.pop(key)
                else:
                    checkpoint[new_key] = checkpoint.pop(key).T
                if "num-tokens" in new_key:
                    del checkpoint[new_key]
    return checkpoint


def get_llm(model_name, cache_dir="llm_weights", local_checkpoint_dir=""):

    if '30b' in model_name or '66b' in model_name:
        model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        cache_dir=cache_dir,
    )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            cache_dir=cache_dir,
            low_cpu_mem_usage=True,
            device_map='auto'
        )

    if local_checkpoint_dir != "":
        checkpoint = torch.load(local_checkpoint_dir, map_location="cpu")
        if "gpt" in model_name:
            checkpoint = convert_flashattn_checkpoint_to_hf(checkpoint)
        model.load_state_dict(checkpoint)

    for i, layer in enumerate(get_layers_list(model)):
        if hasattr(layer, "attn"):
            layer.attn.c_attn = conv1d_to_linear(layer.attn.c_attn)
            layer.attn.c_proj = conv1d_to_linear(layer.attn.c_proj)
        if hasattr(layer, "mlp"):
            layer.mlp.c_fc = conv1d_to_linear(layer.mlp.c_fc)
            layer.mlp.c_proj = conv1d_to_linear(layer.mlp.c_proj)

    model.seqlen = model.config.max_position_embeddings 
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument('--sparsity_ratio', type=float, default=0, help='Sparsity level')
    parser.add_argument("--sparsity_type", type=str, choices=["unstructured", "4:8", "2:4"])
    parser.add_argument("--prune_method", type=str, choices=["magnitude", "wanda", "sparsegpt", 
                        "ablate_mag_seq", "ablate_wanda_seq", "ablate_mag_iter", "ablate_wanda_iter", "search"])
    parser.add_argument("--cache_dir", default="llm_weights", type=str )
    parser.add_argument('--use_variant', action="store_true", help="whether to use the wanda variant described in the appendix")
    parser.add_argument('--save', type=str, default=None, help='Path to save results.')
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')

    parser.add_argument("--eval_zero_shot", action="store_true")

    parser.add_argument("--wanda_in_lora", action="store_true")
    parser.add_argument("--lora_rank", type=float, default=0.0)
    parser.add_argument("--randomized_svd", action="store_true")
    parser.add_argument("--pruned_l", action="store_true")

    parser.add_argument("--bitwidth", type=int, default=8)
    parser.add_argument("--quantization", action="store_true")
    parser.add_argument("--quantize_before_pruning", action="store_true")
    parser.add_argument("--local_checkpoint_dir", type=str, default="")
    parser.add_argument("--eval_dataset", type=str, default="wikitext2", choices=["wikitext2", "c4", "openwebtext"])
    parser.add_argument("--shift_zero_metrics", action="store_true")
    parser.add_argument("--use_std_in_quantization", action="store_true")
    parser.add_argument("--max_bitwidth", type=int, default=8)

    args = parser.parse_args()

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # Handling n:m sparsity
    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured":
        assert args.sparsity_ratio == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))

    model_name = args.model.split("/")[-1]
    print(f"loading llm model {args.model}")
    model = get_llm(args.model, args.cache_dir, args.local_checkpoint_dir)

    single_gpu = False
    if '30b' in args.model or '66b' in args.model:
        single_gpu = True

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    device = torch.device("cuda:0")
    try:
        if "30b" in args.model or "66b" in args.model: # for 30b and 65b we use device_map to load onto multiple A6000 GPUs, thus the processing here.
            device = model.hf_device_map["lm_head"]
    except:
        pass
    print("use device ", device)

    if args.sparsity_ratio != 0:
        print("pruning starts")
        if args.prune_method == "wanda":
            prune_wanda(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m, single_gpu_en = single_gpu)
        elif args.prune_method == "magnitude":
            prune_magnitude(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "sparsegpt":
            prune_sparsegpt(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif "ablate" in args.prune_method:
            prune_ablate(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)

    ################################################################
    print("*"*30)
    sparsity_ratio = check_sparsity(model)
    print(f"sparsity sanity check {sparsity_ratio:.4f}")
    print("*"*30)
    ################################################################
    ppl_test = eval_ppl(args, model, tokenizer, device, single_gpu = single_gpu)
    print(f"wikitext perplexity {ppl_test}")

    if not os.path.exists(args.save):
        os.makedirs(args.save)
    save_filepath = os.path.join(args.save, f"log_{args.prune_method}.txt")
    with open(save_filepath, "w") as f:
        print("method\tactual_sparsity\tppl_test", file=f, flush=True)
        print(f"{args.prune_method}\t{sparsity_ratio:.4f}\t{ppl_test:.4f}", file=f, flush=True)

    if args.eval_zero_shot:
        accelerate=False
        if "30b" in args.model or "66b" in args.model:
            accelerate=True

        task_list = ["boolq", "rte","hellaswag","winogrande", "arc_easy","arc_challenge", "openbookqa"]
        num_shot = 0
        results = eval_zero_shot(args.model, model, tokenizer, task_list, num_shot, accelerate)
        print("********************************")
        print("zero_shot evaluation results")
        print(results)

    if args.save_model:
        model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)

if __name__ == '__main__':
    main()
