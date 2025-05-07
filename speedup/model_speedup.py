from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import argparse
from datasets import load_dataset
import os
import time
import math
import pandas as pd
from model_compression import compress_model


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default='meta-llama/Llama-2-7b-hf')
    parser.add_argument("--seqlen", type=int, help='Sequence length of the model', default=-1)
    parser.add_argument("--max_batch_size", type=int, help='Maximum batch size for inference', default=1)
    parser.add_argument("--iterations", type=int, help='Number of iterations to run', default=3)
    parser.add_argument("--warmup_iters", type=int, help='Number of warmup iterations', default=2)
    parser.add_argument("--single_token_generation", action='store_true',
                        help='Use single token generation')
    parser.add_argument("--time_prefill", action='store_true', help='Time only the prefill step')
    parser.add_argument("--input_token_step", type=int, help='Token step size for input generation',
                        default=1024)
    parser.add_argument("--lora_rank", type=float, help='Rank for LoRA compression', default=0.1)
    parser.add_argument("--compress_model", action='store_true', help='Compress the model')
    parser.add_argument("--quantize_lora", action='store_true', help='Quantize LoRA')
    return parser.parse_args()

def generate_token_chunks(dataset_iter, tokenizer, token_size):
    current_tokens = []
    for example in dataset_iter:
        line = example["text"]
        tokens = tokenizer.encode(line, add_special_tokens=False)
        current_tokens.extend(tokens)
        while len(current_tokens) >= token_size:
            yield torch.tensor(current_tokens[:token_size])
            current_tokens = current_tokens[token_size:]


# Global variable to track current layer index for measuring sdp time
current_layer_idx = None
global_current_data = {}
iter_time = 0


def aggregate_data(csv_path):
    """
    Reads the CSV file, fills missing layer_idx values with 'all' so that model-level rows are preserved,
    computes the median runtime per (component_type, component_name, layer_idx) across iterations,
    then aggregates (sums) over all layers for each (component_type, component_name).
    """
    df = pd.read_csv(csv_path)

    # Fill missing layer_idx values with a placeholder (this preserves model-level rows)
    df['layer_idx'] = df['layer_idx'].fillna('all')

    # Compute median runtime per (component_type, component_name, layer_idx) across iterations
    median_df = df.groupby(['component_type', 'component_name', 'layer_idx'])['time'].median().reset_index()

    # Aggregate over layers. Now model-level rows (with 'all' as layer_idx) are preserved.
    agg_df = median_df.groupby(['component_type', 'component_name'], dropna=False)['time'].sum().reset_index()
    return agg_df


if __name__ == "__main__":
    result_folder = "results/timings"

    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model
    )

    memory_before = torch.cuda.memory_allocated()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        attn_implementation="flash_attention_2",  # Using FlashAttention-2 for faster inference.
        device_map="cpu",
        low_cpu_mem_usage = True
    ).to(torch.float16)

    model.eval()
    if args.compress_model:
        compress_model(model, skip_layers=[model.lm_head], lora_rank=args.lora_rank, quantize_lora=args.quantize_lora)

    model.model.layers = model.model.layers[0:16]

    model = model.cuda()

    memory_after = torch.cuda.memory_allocated()

    # Print allocated memory in GB
    torch.cuda.empty_cache()
    print(f"Allocated memory: {(memory_after - memory_before) / 1e9:.2f} GB")
    device = next(model.parameters()).device

    # Load a text dataset (using wikitext-2 for demonstration) and store data in the specified directory.
    dataset = load_dataset(
        "wikitext",
        "wikitext-2-raw-v1",
        split="train",
    )

    model_seqlen = model.config.max_position_embeddings if args.seqlen == -1 else args.seqlen

    # Benchmark setup
    token_sizes = []
    for k in [args.input_token_step * i for i in range(1, model_seqlen // args.input_token_step)]:
        token_sizes.append(model_seqlen - k)  # Example token sizes
    # Powers of two until args.max_batch_size
    batch_sizes = [2 ** i for i in range(int(min(16, math.log2(args.max_batch_size))), int(math.log2(args.max_batch_size) + 1))]

    # Main benchmarking loop over different token sizes.
    for token_size in token_sizes:
        print(f"\nProcessing token size: {token_size}")
        chunks = list(generate_token_chunks(dataset, tokenizer, token_size))
        print(f"Created {len(chunks)} sequences of {token_size} tokens each.")

        for batch_size in batch_sizes:
            print("\n" + "*" * 80)
            max_new_tokens = model_seqlen - token_size
            print(f"Input Sequence Length: {token_size}, Output Sequence Length: {max_new_tokens}, "
                  f"Model Sequence Length: {model_seqlen}")
            print(f"Batch Size {batch_size}")

            # Prepare batch input
            if len(chunks) < batch_size:
                extended_chunks = chunks * math.ceil(batch_size / len(chunks))
            else:
                extended_chunks = chunks
            batch_input_ids = torch.stack(extended_chunks[:batch_size], dim=0).to(device)


            def model_prehook(module, inputs):
                torch.cuda.synchronize()
                module._start_time = time.time()


            def model_posthook(module, inputs, outputs):
                torch.cuda.synchronize()
                module.cnt += 1
                if args.time_prefill:
                    if module.cnt != 1:
                        return
                else:
                    if module.cnt == 1:
                        return
                global iter_time
                iter_time += time.time() - getattr(module, '_start_time', 0)


            hooks = []
            hooks.append(model.register_forward_pre_hook(model_prehook))
            hooks.append(model.register_forward_hook(model_posthook))
            model.cnt = 0

            # Warmup
            with torch.no_grad():
                for _ in range(args.warmup_iters):
                    if args.single_token_generation:
                        _ = model.generate(
                            input_ids=batch_input_ids,
                            max_new_tokens=token_size,
                            do_sample=True
                        )
                    else:
                        _ = model(input_ids=batch_input_ids)
                    torch.cuda.synchronize()


            # Prepare result folder and CSV for timings
            folder_name = f"{args.model}/batch_{batch_size}_prefill_{token_size}_decode_{model_seqlen - token_size}"
            folder_path = os.path.join(result_folder, folder_name)
            os.makedirs(folder_path, exist_ok=True)
            csv_path = os.path.join(folder_path, "time_decomposition.csv")

            if not os.path.exists(csv_path):
                with open(csv_path, 'w') as f:
                    f.write("batch_size,token_size,iteration,layer_idx,component_type,component_name,time\n")

            with torch.no_grad():
                for it in range(args.iterations):
                    global_current_data.clear()
                    torch.cuda.empty_cache()
                    # start_time = time.time()
                    attention_mask = torch.ones_like(batch_input_ids)
                    iter_time = 0.
                    model.cnt = 0
                    if args.single_token_generation:
                        generated_ids = model.generate(
                            input_ids=batch_input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=max_new_tokens,
                            do_sample=True
                        )
                    else:
                        generated_ids = model(input_ids=batch_input_ids, attention_mask=attention_mask)
                    # torch.cuda.synchronize()
                    # iter_time = time.time() - start_time

                    # Write timings to CSV
                    with open(csv_path, 'a') as f:
                        f.write(f"{batch_size},{token_size},{it},,model_generate,total,{iter_time}\n")
                        for (layer_idx, comp_type, comp_name), t in global_current_data.items():
                            f.write(f"{batch_size},{token_size},{it},{layer_idx},{comp_type},{comp_name},{t}\n")

                    print(f"Iteration {it + 1} completed. Total time: {iter_time:.4f}s")

            # Cleanup hooks and free memory
            for hook in hooks:
                hook.remove()
            del batch_input_ids
            torch.cuda.empty_cache()
