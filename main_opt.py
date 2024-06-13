import argparse
import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, LlamaTokenizer
from importlib.metadata import version

from lib.prune_opt import prune_wanda, prune_magnitude, prune_sparsegpt, prune_ablate, check_sparsity, quantize_model
from lib.eval import eval_ppl, eval_zero_shot
from lib.utils import get_layers_list

CSV_COLUMNS = ["model", "prune_method", "sparsity_ratio", "sparsity_type", "lora_rank",
               "wanda_in_lora", "randomized_svd", "shift_zero_metrics", "eval_dataset", "quantize",
               "quantize_before_pruning",
               "bitwidth", "max_bitwidth", "use_std_in_quantization", "bias_correction", "bias_alpha",
               "bias_correction_nsamples", "perplexity", "mmlu"]

print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())

hf_token = "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"


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
                new_key = new_key.replace("fc1", "c_fc")
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
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        cache_dir=cache_dir,
        low_cpu_mem_usage=True,
        token=hf_token,
        # device_map='auto'
    )
    layer_num_params = 0
    for param in model.parameters():
        layer_num_params += param.numel()
    model_size = layer_num_params * 2
    free_mem, total_mem = torch.cuda.mem_get_info()
    if model_size < 0.75 * free_mem:
        print("Loading model in GPU...")
        model = model.cuda()
    else:
        print("Model does not fit in GPUs. Loading model in CPU...")

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
            if hasattr(layer.mlp, "c_fc"):
                layer.mlp.c_fc = conv1d_to_linear(layer.mlp.c_fc)
            if hasattr(layer.mlp, "c_proj"):
                layer.mlp.c_proj = conv1d_to_linear(layer.mlp.c_proj)

    model.seqlen = model.config.max_position_embeddings
    return model


def add_result_to_csv(args, ppl, mmlu):
    # Load CSV if it exists, otherwise create a new DataFrame with given columns
    if os.path.exists(args.output_csv_path):
        df = pd.read_csv(args.output_csv_path)
    else:
        df = pd.DataFrame(columns=CSV_COLUMNS)

    # Check if the row combination exists and update perplexity
    new_row_data = {column: getattr(args, column) for column in CSV_COLUMNS[:-2]}
    row_exists = df.index[(df[CSV_COLUMNS[:-2]] == pd.Series(new_row_data)).all(axis=1)]

    # Now we don't mind adding perplexity
    new_row_data['perplexity'] = ppl
    new_row_data['mmlu'] = mmlu

    if row_exists.empty:
        # Row combination does not exist, add a new row
        new_row_df = pd.DataFrame([new_row_data], columns=CSV_COLUMNS)
        df = pd.concat([df, new_row_df], ignore_index=True)
    else:
        # Row combination exists, modify perplexity
        index_to_update = row_exists.values[0]
        df.at[index_to_update, 'perplexity'] = new_row_data['perplexity']
        df.at[index_to_update, 'mmlu'] = mmlu

    # Save to CSV
    df.to_csv(args.output_csv_path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument('--sparsity_ratio', type=float, default=0, help='Sparsity level')
    parser.add_argument("--sparsity_type", type=str)
    parser.add_argument("--prune_method", type=str, choices=["magnitude", "wanda", "sparsegpt",
                                                             "ablate_mag_seq", "ablate_wanda_seq", "ablate_mag_iter",
                                                             "ablate_wanda_iter", "search"])
    parser.add_argument("--cache_dir", default="llm_weights", type=str)
    parser.add_argument('--use_variant', action="store_true",
                        help="whether to use the wanda variant described in the appendix")
    parser.add_argument('--save', type=str, default=None, help='Path to save results.')
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')

    parser.add_argument("--eval_zero_shot", action="store_true")
    parser.add_argument('--num_sample_partition', type=int, default=8,
                        help='Number of partitions for evaluation samples.')

    parser.add_argument("--wanda_in_lora", action="store_true")
    parser.add_argument("--lora_rank", type=float, default=0.0)
    parser.add_argument("--separate_lora", action="store_true")
    parser.add_argument("--randomized_svd", action="store_true")
    parser.add_argument("--pruned_l", action="store_true")
    parser.add_argument("--bias_correction", action="store_true")
    parser.add_argument("--bias_alpha", type=float, default=1.0)
    parser.add_argument("--bias_correction_nsamples", type=int, default=128)

    parser.add_argument("--bitwidth", type=int, default=8)
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--quantize_before_pruning", action="store_true")
    parser.add_argument("--local_checkpoint_dir", type=str, default="")
    parser.add_argument("--eval_dataset", type=str, default="wikitext2", choices=["wikitext2", "c4", "openwebtext"])
    parser.add_argument("--shift_zero_metrics", action="store_true")
    parser.add_argument("--use_std_in_quantization", action="store_true")
    parser.add_argument("--max_bitwidth", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--output_csv_path", type=str, default=None, help='Output CSV to accumulate experiment result')
    parser.add_argument('--accelerate', action="store_true", help="Whether to use cuSparseLt backend")
    parser.add_argument('--test_mmlu', action="store_true", help="Whether to test mmlu")

    args = parser.parse_args()

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    model_name = args.model.split("/")[-1]
    print(f"Loading model {model_name}")
    model = get_llm(args.model, args.cache_dir, args.local_checkpoint_dir)

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False, cache_dir="tokenizers", token=hf_token)

    device = torch.device("cuda:0")
    print("Device ", device)
    print("Sparsity Ratio: ", args.sparsity_ratio)
    print("Pruning Structure: ", args.sparsity_type)
    print("Prune Method: ", args.prune_method)
    print("Quantize: ", args.quantize)

    if args.sparsity_ratio == 0. or args.sparsity_type == "dense":
        if args.quantize:
            print("Quantizing the dense model:")
            quantize_model(args, model)
        else:
            print("Using original dense model:")
    else:
        # Handling n:m sparsity
        prune_n, prune_m = 0, 0
        if args.sparsity_type != "unstructured":
            prune_n, prune_m = map(int, args.sparsity_type.split(":"))
            prune_n = prune_m - prune_n
            assert args.sparsity_ratio == prune_n / prune_m, "sparsity ratio must be 0.5 for structured N:M sparsity"
        print(f"Pruning and quantizing the model using {args.prune_method}:")
        if args.prune_method == "wanda":
            prune_wanda(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "magnitude":
            prune_magnitude(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "sparsegpt":
            prune_sparsegpt(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif "ablate" in args.prune_method:
            prune_ablate(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)

    ################################################################
    print("*" * 30)
    sparsity_ratio = check_sparsity(model)
    print(f"Model Sparsity Ratio: {sparsity_ratio:.4f}")
    print("*" * 30)
    ################################################################
    ppl_test = eval_ppl(args, model, tokenizer, device, num_partition = args.num_sample_partition)
    print(f"WikiText2 Perplexity: {ppl_test}")


    if args.test_mmlu:
        import lm_eval
        randint = np.random.randint(0, 1000)
        checkpoint_dir = "llm_weights/tmp_ckpt{}".format(randint)
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        results = lm_eval.simple_evaluate(
            model="hf",
            model_args=f"pretrained={checkpoint_dir},dtype=half,parallelize=True,device_map_option=auto",
            tasks="mmlu",
            verbosity="ERROR"
        )
        mmlu = results['results']["mmlu"]["acc,none"]
        print("MMLU: ", mmlu)

    if args.output_csv_path:
        add_result_to_csv(args, ppl_test, mmlu)

    if not os.path.exists(args.save):
        os.makedirs(args.save)
    save_filepath = os.path.join(args.save, f"log_{args.prune_method}.txt")
    with open(save_filepath, "w") as f:
        print("method\tactual_sparsity\tppl_test", file=f, flush=True)
        print(f"{args.prune_method}\t{sparsity_ratio:.4f}\t{ppl_test:.4f}", file=f, flush=True)

    if args.eval_zero_shot:
        accelerate = False
        if "30b" in args.model or "66b" in args.model:
            accelerate = True

        task_list = ["boolq", "rte", "hellaswag", "winogrande", "arc_easy", "arc_challenge", "openbookqa"]
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
