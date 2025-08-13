from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os
import safetensors.torch
import torch
import numpy as np
from tqdm import tqdm
from slim.utils import get_layers_list, find_layers
from slim.quantization.quantization import Quantizer as AutoQuantizer, dequantize_tensor
from utils.model import add_empty_lora
import torch.nn as nn

MODEL_PATH = '/home/hungshou/links/scratch/mohammad/slim-dev/checkpoints/llama3.1_8B_wanda_2:4_lr0.1_sparsity0.5'

def load_compressed_model(model_path):
    # Load configuration from args.json
    with open(os.path.join(model_path, 'args.json')) as f:
        args = json.load(f)
    
    print("Loading model configuration...")
    print(f"Model: {args['model']}")
    print(f"Pruning method: {args['prune_method']}")
    print(f"Sparsity ratio: {args['sparsity_ratio']}")
    print(f"Sparsity type: {args['sparsity_type']}")
    print(f"Quantize weights: {args['quantize_weight']}")
    print(f"LoRA rank: {args['lora_rank']}")
    print(f"Separate LoRA: {args['separate_lora']}")
    print(f"Quantize LoRA: {args['quantize_lora']}")
    
    # Load base model arch
    print("Loading base model architecture...")
    model = AutoModelForCausalLM.from_pretrained(
        args['model'], 
        torch_dtype=torch.bfloat16,
        device_map='cpu'  # Load onto cpu first
    )
    
    # Determine if we need LoRA (simple check: lora_rank > 0)
    has_lora = args.get('lora_rank', 0) > 0
    
    if has_lora:
        # Note that add_empty_lora already attaches the hooks
        print("Adding LoRA architecture...")
        lora_tile_size = args['lora_tile_size'] if args.get('quantize_lora', False) else None
        add_empty_lora(
            model,
            lora_tile_size=lora_tile_size,
            lora_rank=args['lora_rank']
        )
    # Load the full state dict
    print("Loading state dictionary...")
    final_state_dict = {}
    
    # Use model.safetensors.index.json to figure out the filenames
    with open(os.path.join(model_path, 'model.safetensors.index.json')) as f:
        index_data = json.load(f)
        filenames = set()
        for file in index_data['weight_map'].values():
            filenames.add(file)
    
    files = list(filenames)
    for file in (pbar := tqdm(files, desc="Loading state dict files")):
        pbar.set_description_str(f"Loading file {file}")
        load_files_dict = safetensors.torch.load_file(os.path.join(model_path, file))
        final_state_dict.update(load_files_dict)
    
    # Load the state dict into the model
    print("Loading state dict into model...")
    matched_keys, unmatched_keys = model.load_state_dict(final_state_dict, strict=False)
    
    if unmatched_keys:
        # We are saving extra scaling factors, but they are not needed, 
        # as we have already dequantized the weights and stored them in both lora and the model itself
        print('Unmatched keys detected, showing top 5 for debugging purposes')
        print(unmatched_keys[:5])
    
    # Apply input quantization hooks if they were used
    if args.get('quantize_input', False):
        print("Setting up input quantization hooks...")
        from slim.quantization.quantization import attach_input_quantization_hooks
        attach_input_quantization_hooks(model,
                                        args['input_bitwidth'],
                                        args['input_group_size'])
    
    print("Model loaded successfully!")
    return model, args

# Load the model
model, cfg = load_compressed_model(MODEL_PATH)

try:
    # Running eval again
    
    from transformers import AutoTokenizer
    from slim.eval import eval_ppl_wikitext
    from slim.data import get_wikitext2
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg['model'], 
        local_files_only=cfg.get('local_files_only', False)
    )
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Setting up model for evaluation...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    
    model.config.max_position_embeddings = 2048  # Set to eval ppl sequences with 2048
    
    print(f"Model moved to device: {device}")
    print(f"Model dtype: {next(model.parameters()).dtype}")
    
    # Load WikiText2 test data
    print("Loading WikiText2 test dataset...")
    _, testenc = get_wikitext2(seed=0, tokenizer=tokenizer)
    
    print(f"Test dataset size: {testenc.input_ids.numel()} tokens")
    
    # Evaluate perplexity
    print("Evaluating perplexity on WikiText2...")
    ppl = eval_ppl_wikitext(
        model=model,
        testenc=testenc,
        bs=cfg.get('eval_batch_size', 1),
        device=device
    )
    
    print(f"WikiText2 Perplexity: {ppl:.2f}")
    
except Exception as e:
    print(f"Error during perplexity evaluation: {e}")
    import traceback
    traceback.print_exc()