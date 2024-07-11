import argparse
import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, LlamaTokenizer
from importlib.metadata import version

from lib.prune_opt import prune_wanda, prune_magnitude, prune_sparsegpt, prune_ablate, check_sparsity, quantize_model
from lib.eval import eval_ppl, eval_zero_shot
from lib.utils import contigous_model, merge_lora, get_llm, hf_token, convert_linear_to_conv1d
import time
import shutil
from lib.fine_tune import fine_tune


CSV_COLUMNS = ["model", "prune_method", "sparsity_ratio", "sparsity_type", "lora_rank",
               "wanda_in_lora", "randomized_svd", "shift_zero_metrics", "eval_dataset", "quantize",
               "quantize_before_pruning",
               "bitwidth", "max_bitwidth", "use_std_in_quantization", "bias_correction", "bias_alpha",
               "bias_correction_nsamples", "perplexity", "mmlu", "piqa", "arc_easy", "arc_challenge",
               "winogrande", "openbookqa", "average"]

print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())




def add_result_to_csv(args, ppl, lmharness_results):
    # Load CSV if it exists, otherwise create a new DataFrame with given columns
    if os.path.exists(args.output_csv_path):
        df = pd.read_csv(args.output_csv_path)
    else:
        df = pd.DataFrame(columns=CSV_COLUMNS)

    num_tasks = 8

    # Check if the row combination exists and update perplexity
    new_row_data = {column: getattr(args, column) for column in CSV_COLUMNS[:-num_tasks]}
    row_exists = df.index[(df[CSV_COLUMNS[:-num_tasks]] == pd.Series(new_row_data)).all(axis=1)]

    # Now we don't mind adding perplexity
    new_row_data['perplexity'] = ppl
    for task in lmharness_results:
        new_row_data[task] = lmharness_results[task]

    if row_exists.empty:
        # Row combination does not exist, add a new row
        new_row_df = pd.DataFrame([new_row_data], columns=CSV_COLUMNS)
        df = pd.concat([df, new_row_df], ignore_index=True)
    else:
        # Row combination exists, modify perplexity
        index_to_update = row_exists.values[0]
        df.at[index_to_update, 'perplexity'] = new_row_data['perplexity']
        for task in lmharness_results:
            df.at[index_to_update, task] = lmharness_results[task]

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
    parser.add_argument('--test_lmharness', action="store_true", help="Whether to test LMEHarness tasks")
    parser.add_argument('--fine_tune', action="store_true", help="Whether to fine-tune the model after pruning")
    parser.add_argument('--evaluate_perplexity', action="store_true", help="Whether to evaluate the model perplexity")
    parser.add_argument('--local_files_only', action="store_true", help="Whether to use local files only")

    args = parser.parse_args()

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    model_name = args.model.split("/")[-1]
    print(f"Loading model {model_name}")
    model = get_llm(args.model, args.cache_dir, args.local_checkpoint_dir, local_files_only=args.local_files_only)

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
    if args.fine_tune:
        fine_tune(model, tokenizer)#, block_size=tokenizer.model_max_length)
        print("*" * 30)

    ################################################################
    ppl_test = 0.
    if args.evaluate_perplexity:
        ppl_test = eval_ppl(args, model, tokenizer, args.model,  device, num_partition=args.num_sample_partition)
        print(f"WikiText2 Perplexity: {ppl_test}")
        print("*" * 30)
    ################################################################

    merge_lora(model)

    
    lmharness_results = {}
    if args.test_lmharness:
        import lm_eval
        np.random.seed(np.int64(time.time()))
        randint = np.random.randint(0, 1000)
        checkpoint_dir = "/tmp/tmp_ckpt{}".format(randint)
        if model.has_conv1d:
            convert_linear_to_conv1d(model)
        contigous_model(model)
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        del model, tokenizer
        torch.cuda.empty_cache()
        results = lm_eval.simple_evaluate(
            model="hf",
            model_args=f"pretrained={checkpoint_dir},dtype=half,parallelize=True,device_map_option=auto",
            tasks=["mmlu", "piqa", "arc_easy", "arc_challenge", "winogrande", "openbookqa"],
            verbosity="ERROR"
        )
        shutil.rmtree(checkpoint_dir)
        lmharness_results["mmlu"] = results['results']["mmlu"]["acc,none"]
        lmharness_results["piqa"] = results['results']["piqa"]["acc,none"]
        lmharness_results["arc_easy"] = results['results']["arc_easy"]["acc,none"]
        lmharness_results["arc_challenge"] = results['results']["arc_challenge"]["acc,none"]
        lmharness_results["winogrande"] = results['results']["winogrande"]["acc,none"]
        lmharness_results["openbookqa"] = results['results']["openbookqa"]["acc,none"]
        average = []
        for task in lmharness_results:
            average.append(lmharness_results[task])
        average = np.mean(average)
        lmharness_results["average"] = average
        print("LM Harness Results: ", lmharness_results)

        

    if args.output_csv_path:
        add_result_to_csv(args, ppl_test, lmharness_results)

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
