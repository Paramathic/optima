# SLiM: One-shot Quantized Sparse Plus Low-rank Approximation of LLMs

This repository contains the implementation of SLiM (Sparse Low-rank Approximation with Quantization), a novel 
compression technique for large language models (LLMs). SLiM combines a one-shot quantization and sparse low-rank 
approximation to reduce memory usage and improve inference speed without requiring retraining. The approach features 
SLIM-Quant, a symmetric quantization method, and a saliency-based low-rank approximation that leverages sparsity 
patterns like 2:4 for optimized performance on accelerated hardware. With this, SLiM offers state-of-the-art accuracy 
while maintaining efficiency in memory-constrained environments.

**SLiM: One-shot Quantized Sparse Plus Low-rank Approximation of LLMs**

*Mohammad Mozaffari and Maryam Mehri Dehnavi*

Paper: TODO: Add arXiv link

![Alt text](./assets/SLiM-Pipeline.png "MKOR Pipeline")

## Setup

The list of requirements can be found in the `requirements.txt` file. To install the requirements, run the following command:

```bash 
pip install -r requirements.txt
```

## Quick Start
Our code base supports multiple pruning, quantization, and low-rank approximation techniques. Below, we provide an 
example and a brief description of how to use our code base. For a more automated and detailed example, please refer to
[srcipts/run.sh](scripts/run.sh).

**Model and Tokenizer Instantiation:** Our code base supports models from HuggingFace's transformers library. In this example, we use
the OPT-125M model from [facebook/opt-125m](https://huggingface.co/facebook/opt-125m).

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


model_name = "facebook/opt-125m"

model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).cuda()

model.eval()

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    use_fast=False,
    cache_dir="tokenizers",
)
```

**Compression:** We provide a function `prune_and_quantize` that takes in a model, tokenizer, and depending on the 
input arguments prunes, quantizes, and add low-rank approximation to the model. Below, we provide an example of how to 
use it for SLiM Low-rank approximation and SLiM-Quant quantization method. More details about the `prune_and_quantize`
function are provided in the **Function Documentation** section.

```python
from slim.prune import prune_and_quantize

quantize_lora_flag = True
lora_tile_size = 256
quantization_bitwidth = 4

prune_and_quantize(
    model=model,
    tokenizer=tokenizer,
    prune_method="wanda",
    sparsity_ratio=0.5,
    quantize_weight=False,
    bitwidth=quantization_bitwidth,
    slim_quant=True,
    lora_rank=0.1,
    sparsity_type="2:4",
    weight_tiled_quantization=quantize_lora_flag,
    quantize_lora=lora_tile_size,
)
```

**Optional Fine-tuning:** After compression, the model can be fine-tuned to compensate for the accuracy loss using the
`fine_tune` function provided in our code base. When using low-rank adapters, the `fine_tune` function will 
automatically freeze the original weights and biases and only fine-tunes the low-rank adapters. Otherwise, the original
weights and biases will be fine-tuned and requantized in the end if needed. Below, we provide an example of how to use
the `fine_tune` function. More details about the `fine_tune` function are provided in the **Function Documentation** section.

```python
from slim.fine_tune import fine_tune

fine_tune(
    model,
    tokenizer,
    max_train_samples=30000,
    optimizer="adafactor",
    global_batch_size=64,
    local_batch_size=8,
)
```

**Adapter Quantization:** In case the `quantize_lora` is set to `True` in the `prune_and_quantize` function, the low-rank
will be prepared for quantization. To finalize the adapter quantization, you can use the `quantize_lora` function.

```python
from slim.lora import quantize_lora

if quantize_lora_flag:
    quantize_lora(
        model,
        bitwidth=quantization_bitwidth,
        lora_tile_size=lora_tile_size,
    )
```

**Input Quantization:** You can emulate input group quantization using the `attach_input_quantization_hooks` function. 
This function attaches hooks to the linear layers of the model to quantize the input activations. The function will
skip the quantization of the input of the layer if the input quantization error surpasses 5%. Please note that
input quantization is only supported for 1-dimensional group quantization using AbsMax and works well with 8 bits.

```python
from slim.quantization import attach_input_quantization_hooks

attach_input_quantization_hooks(
    model,
    bitwidth=8,
    input_group_size=128,
)
```

**Check Sparsity Ratio:** You can check the sparsity ratio of the model using the `check_sparsity` function.

```python
from slim.utils import check_sparsity

check_sparsity(model)
```

**Evaluate Perplexity:** You can evaluate the perplexity of the model using the `eval_ppl` function.

```python
from slim.eval import eval_ppl

ppl_test = eval_ppl(
    model,
    tokenizer,
    eval_dataset="wikitext2",
    eval_batch_size=8,
)

print(f"WikiText2 Perplexity: {ppl_test:.2f}")
```

**Zero-shot Task Evaluation:** For running the zero-shot task evaluation on the model, and a more automated example of
using the code base, please refer to the [scripts/run.sh](scripts/run.sh) file. You can run it by executing the 
following command.

```bash
bash scripts/run.sh
```


## Experimental Results

We provide extensive experimental results in the paper. For completeness, we have provided the average accuracy results 
of sparse and quantized models on a range of zero-shot tasks using different pruning and quantization methods in the 
table below. The weights (and possibly the adapter) are quantize to 4 bits using symmetric quantization, and the inputs 
are quantized using 8-bit group quantization. All the group quantization results use a group size of 128.

| Pruning Method | Weight Quantization | 125M | 350M | 1.3B | 2.7B | 6.7B | 13B | 7B | 13B |
|----------------|---------------------|------|------|------|------|------|-----|-----|-----|
| Dense          | -                   | 35.9 | 37.1 | 43.4 | 45.5 | 48.3 | 48.7 | 56.6 | 60.8 |
| **50% 2:4**    |                     |      |      |      |      |      |     |     |     |
| Magnitude      | AbsMax              | 32.0 | 31.8 | 34.2 | 32.5 | 35.3 | 30.8 | 31.2 | 32.1 |
| SparseGPT      | Group-OPTQ          | 33.7 | 32.6 | 37.3 | 40.2 | 44.4 | 45.5 | 45.4 | 50.8 |
| SparseGPT      | OPTQ                | 31.4 | 32.9 | 31.0 | 33.9 | 39.9 | 40.0 | 31.8 | 31.6 |
| Wanda          | Group AbsMax        | 33.0 | 31.6 | 36.3 | 35.1 | 36.6 | 43.4 | 43.1 | 48.3 |
| Wanda          | AbsMax              | 31.5 | 31.3 | 31.6 | 30.7 | 30.5 | 31.2 | 32.0 | 31.3 |
| Wanda          | SLiM-Quant          | 31.8 | 32.1 | 34.7 | 34.3 | 38.4 | 32.8 | 30.8 | 30.7 |
| Wanda-SVD      | Group AbsMax        | 33.9 | 34.0 | 38.9 | 39.9 | 44.2 | 45.5 | 50.5 | 54.5 |
| Wanda-SVD      | SLiM-Quant          | 34.2 | 33.3 | 38.7 | 41.2 | 44.3 | 45.2 | 48.3 | 51.4 |
| Wanda-SVD + FT | SLiM-Quant          | 34.0 | 34.3 | 39.6 | 42.6 | **46.1** | 47.2 | 50.8 | 55.4 |
| SLiM           | Group AbsMax        | 33.9 | 33.7 | 39.9 | 42.8 | 45.8 | 46.0 | 50.2 | 54.3 |
| SLiM           | SLiM-Quant          | 34.3 | 33.5 | 40.0 | 42.8 | **46.1** | 46.1 | **50.8** | 54.8 |
| SLiM$^Q$       | SLiM-Quant          | 34.2 | 33.8 | 39.8 | 41.8 | 46.0 | 45.9 | 50.6 | 53.0 |
| SLiM  + FT     | SLiM-Quant          | **34.9** | **34.5** | **41.3** | **43.5** | **46.1** | **47.3** | 50.5 | **56.6** |
| SLiM$^Q$ + FT  | SLiM-Quant          | **34.9** | 34.3 | 40.0 | 42.3 | 46.0 | 46.5 | 50.6 | 54.1 |
| **50% Unstructured** |               |      |      |      |      |      |     |     |     |
| Magnitude      | AbsMax              | 31.1 | 32.9 | 33.1 | 36.2 | 36.3 | 31.2 | 32.6 | 31.5 |
| SparseGPT      | Group-OPTQ          | 35.1 | 35.1 | 38.9 | 43.2 | 47.1 | 47.3 | 50.1 | 55.4 |
| SparseGPT      | OPTQ                | 31.4 | 34.5 | 31.2 | 37.1 | 43.2 | 44.1 | 31.7 | 32.0 |
| Wanda          | Group AbsMax        | 34.2 | 33.3 | 39.1 | 40.7 | 44.9 | 46.2 | 51.7 | 55.8 |
| Wanda          | AbsMax              | 31.5 | 32.9 | 31.0 | 32.9 | 30.5 | 31.1 | 32.7 | 31.1 |
| Wanda          | SLiM-Quant          | 32.8 | 33.9 | 36.0 | 36.2 | 42.7 | 32.8 | 30.4 | 30.5 |
| Wanda-SVD      | Group AbsMax        | 34.6 | 34.4 | 40.5 | 42.9 | 46.3 | 47.2 | 53.9 | 55.4 |
| Wanda-SVD      | SLiM-Quant          | 34.6 | 34.4 | 40.3 | 43.3 | 46.7 | 45.2 | 51.2 | 55.4 |
| Wanda-SVD + FT | SLiM-Quant          | 35.3 | 34.8 | 41.8 | 43.8 | 47.0 | 47.9 | 53.0 | 57.3 |
| SLiM           | Group AbsMax        | 35.0 | 35.0 | 41.5 | 43.6 | 47.2 | 47.9 | **54.0** | **57.6** |
| SLiM           | SLiM-Quant          | **35.7** | 35.4 | 42.0 | 43.4 | 47.5 | 48.0 | **54.0** | **57.6** |
| SLiM$^Q$       | SLiM-Quant          | 34.8 | 35.0 | 41.4 | 34.3 | 47.1 | 47.4 | 53.8 | 57.1 |
| SLiM  + FT     | SLiM-Quant          | **35.7** | **35.8** | **42.3** | **44.3** | 47.3 | **48.4** | 53.2 | 57.0 |
| SLiM$^Q$ + FT  | SLiM-Quant          | 35.3 | 35.6 | 41.9 | 43.8 | **47.6** | 48.1 | 53.7 | **57.6** |


## Function Documentation
Here we provide a brief description of a few of the main functions in our code base. For details about the other 
functions, please refer to their dockstrings.
### **lib.prune.prune_and_quantize:**
- `model`: The model to be pruned and quantized.
- `tokenizer`: The tokenizer of the model.
- `bitwidth`: The bitwidth to be used for quantization.
- `slim_quant`: Whether to use SLiM-Quant for pruning. If set to 'False', AbsMax or OPTQ (GPTQ) will be used for quantization.
- `weight_tiled_quantization`: Whether to use weight tiled (group) quantization. We do not recommend using this option with SLiM-Quant.
- `weight_tile_size`: The size of the weight tiles to be used for weight tiled quantization. The dimension of the tile will be $\sqrt{\text{weight_tile_size}}$.
- `prune_method`: The pruning method to be used. We support `wanda`, `sparsegpt`, and `magnitude`. If using `sparsegpt`, the `slim_quant` should be set to `False`.
- `sparsity_ratio`: The sparsity ratio to be used for pruning.
- `sparsity_type`: The sparsity type to be used for pruning. We support `unstructured` and `N:M` sparsity.
- `quantize_weight`: Whether to quantize the weights of the model.
- `nsamples`: The number of samples for calibration.
- `shift_zero_metrics`: Whether to shift the zero metrics in Wanda.
- `lora_rank`: The rank to be used for low-rank approximation (between 0. and 1.). If set to 0., no low-rank approximation will be used.
- `slim_lora`: Whether to use SLiM for low-rank approximation.
- `prune_lora`: Whether to 2:4 prune the left low-rank adapter `L`. For setting this option, `sparsity_type` should be set to `2:4`.
- `quantize_lora`: Whether to quantize the low-rank adapters.
- `lora_tile_size`: The size of the low-rank adapter tiles to be used for low-rank approximation. The dimension of the tile will be $\sqrt{\text{lora_tile_size}}$.
- `separate_lora`: Whether to keep the low-rank adapters separate from the model weights. If set to `False`, the low-rank adapters will be merged with the model weights.
- `seed`: The seed to be used for reproducibility.

### **lib.fine_tune.fine_tune:**
- `model`: The model to be fine-tuned.
- `tokenizer`: The tokenizer of the model.
- `dataset_name`: The dataset to be used for fine-tuning.
- `dataset_config_name`: The configuration of the dataset to be used for fine-tuning.
- `validation_split_percentage`: The percentage of the dataset to be used for validation.
- `streaming`: Whether to use streaming for loading the dataset.
- `preprocessing_num_workers`: The number of workers to be used for preprocessing.
- `overwrite_cache`: Whether to overwrite the cache.
- `block_size`: The block size to be used for the dataset.
- `max_train_samples`: The maximum number of samples to be used for training.
- `max_eval_samples`: The maximum number of samples to be used for evaluation.
- `cache_dir`: The directory to be used for caching.
- `optimizer`: The optimizer to be used for fine-tuning. We suggest using `adamw_torch` for as the optimizer. In case low avaiable memory, `adafactor` can be used.
- `global_batch_size`: The global batch size to be used for fine-tuning.
- `local_batch_size`: The local batch size to be used for fine-tuning.

## Acknowledgement
This repository is build upon the [SparseGPT](https://github.com/IST-DASLab/sparsegpt) and the [Wanda](https://github.com/locuslab/wanda) repository.

## Citation
If you use SLiM in your research, please cite our paper:
```angular2html
TODO: Add citation
```