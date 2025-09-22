# OPTIMA: Optimal One-Shot Pruning for LLMs via Quadratic Programming Reconstruction

This repository contains the implementation of OPTIMA, a practical one-shot post-training pruning method for large language models (LLMs). OPTIMA reformulates layer-wise weight reconstruction as independent, row-wise Quadratic Programs (QPs) that share a common layer Hessian, enabling globally optimal updates with respect to the reconstruction objective. It integrates with existing mask selectors (e.g., Wanda, SparseGPT, Thanos) and is designed for accelerator-friendly execution, balancing accuracy and scalability without fine-tuning.

**OPTIMA: Optimal One-Shot Pruning for LLMs via Quadratic Programming Reconstruction**

<img src="./assets/OPTIMA-Logo.png" alt="OPTIMA" width="400">  

## Setup

The list of requirements can be found in the `requirements.txt` file. To install the requirements, run the following command:

```bash 
pip install -r requirements.txt
```

## Quick Start
Our code base supports multiple pruning methods with OPTIMA's optimal weight updates via Quadratic Programming. Below, we provide an 
example and a brief description of how to use our code base. For a more automated and detailed example, please refer to
[srcipts/run.sh](scripts/run.sh).

**Model and Tokenizer Instantiation:** Our code base supports models from HuggingFace's transformers library. In this example, we use
the OPT-125M model from [facebook/opt-125m](https://huggingface.co/facebook/opt-125m). Please note that we load the model in CPU to reduce memory overheads 
on GPUs. Our code supports single-GPU compression of very large models, as long as a single transformer block of the 
model fits in the GPU memory.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


model_name = "facebook/opt-125m"

model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

model.eval()

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    use_fast=False,
)
```

**Compression:** We provide a function `prune_and_quantize` that takes in a model, tokenizer, and depending on the 
input arguments prunes, quantizes, and add low-rank approximation to the model. Below, we provide an example of how to 
use it for SLiM Low-rank approximation and SLiM-Quant quantization method. More details about the `prune_and_quantize`
function are provided in the **Function Documentation** section.

```python
from optima.prune import prune_and_quantize

prune_and_quantize(
    model=model,
    tokenizer=tokenizer,
    prune_method="wanda",
    sparsity_ratio=0.5,
    sparsity_type="2:4",
    update_weights=True,
    use_qp_solver=True,
    double_precision=False,
    skip_attention=True,
)
```


**Check Sparsity Ratio:** You can check the sparsity ratio of the model using the `check_sparsity` function.

```python
from slim_local.slim.utils import check_sparsity

check_sparsity(model)
```

**Evaluate Perplexity:** You can evaluate the perplexity of the model using the `eval_ppl` function.

```python
from slim_local.slim.eval import eval_ppl

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

For scheduling jobs on a cluster, you can use the [scripts/submit_jobs.sh](scripts/submit_jobs.sh) file. Please note that you need to

**Note:** If your cluster does not have internet access, you can download the models and datasets using the `slim_local/scripts/download_data.sh` script.

## Experimental Results

We provide extensive experimental results in the paper. For completeness, we have included results from Table 1 of the paper, showing model perplexity on WikiText2 and average accuracy on zero-shot downstream tasks (MMLU, PIQA, Arc-E, Arc-C, Wino, OpenQA) for 50% unstructured sparsity. OPTIMA is applied as a weight update mechanism on top of mask selectors like Wanda, SparseGPT, and Thanos.

Notes:
- Dense refers to the unpruned baseline.
- OPTIMA consistently improves accuracy across models and tasks.
- **Bold values** indicate the best performance per model (excluding dense).

## Model Perplexity and Average Accuracy on Zero-shot Tasks for 50% Unstructured Sparsity


| Model         | Mask Selection | Weight Update | Perplexity | Avg. Accuracy (%) |
|---------------|----------------|---------------|------------|-------------------|
| **LLaMA 3.2 1B** | Dense      | -             | 9.75       | 49.09             |
|               | Wanda      | -             | 23.51      | 40.01             |
|               | Wanda      | OPTIMA        | 18.84      | **41.33**             |
|               | SparseGPT  | SparseGPT     | 18.84      | 42.35             |
|               | SparseGPT  | OPTIMA        | 18.09      | **42.72**             |
|               | Thanos     | Thanos        | 19.70      | 41.62             |
|               | Thanos     | OPTIMA        | 18.77      | **41.94**             |
| **LLaMA 3.2 3B** | Dense      | -             | 7.81       | 57.95             |
|               | Wanda      | -             | 12.92      | 49.95             |
|               | Wanda      | OPTIMA        | 12.24      | **51.37**             |
|               | SparseGPT  | SparseGPT     | 12.32      | 50.20             |
|               | SparseGPT  | OPTIMA        | 12.43      | **51.39**             |
|               | Thanos     | Thanos        | 12.26      | 50.81             |
|               | Thanos     | OPTIMA        | 12.40      | **51.41**             |
| **Gemma 3 1B**  | Dense      | -             | 14.17      | 49.10             |
|               | Wanda      | -             | 32.96      | 42.21             |
|               | Wanda      | OPTIMA        | 28.90      | **44.01**             |
|               | SparseGPT  | SparseGPT     | 28.34      | 43.03             |
|               | SparseGPT  | OPTIMA        | 27.35      | **43.76**             |
|               | Thanos     | Thanos        | 28.65      | 43.88             |
|               | Thanos     | OPTIMA        | 28.14      | **44.05**             |
| **Gemma 2 2B**  | Dense      | -             | 68.69      | 59.16             |
|               | Wanda      | -             | 327.45     | **50.27**             |
|               | Wanda      | OPTIMA        | 215.63     | 50.10             |
|               | SparseGPT  | SparseGPT     | 234.68     | 51.24             |
|               | SparseGPT  | OPTIMA        | 241.09     | **51.60**             |
|               | Thanos     | Thanos        | 276.97     | 49.19             |
|               | Thanos     | OPTIMA        | 250.15     | **49.94**             |
| **LLaMA 3.1 8B** | Dense      | -             | 5.84       | 63.89             |
|               | Wanda      | -             | 9.64       | 55.70             |
|               | Wanda      | OPTIMA        | 9.37       | **56.70**             |
|               | SparseGPT  | SparseGPT     | 9.30       | 57.01             |
|               | SparseGPT  | OPTIMA        | 9.33       | **57.02**             |
|               | Thanos     | Thanos        | 9.27       | **57.64**             |
|               | Thanos     | OPTIMA        | 9.35       | 56.89             |


## Function Documentation
Here we provide a brief description of a few of the main functions in our code base. For details about the other 
functions, please refer to their dockstrings.
### **optima.prune.prune_and_quantize:**
- `model`: The model to be pruned and quantized.
- `tokenizer`: The tokenizer of the model.
- `bitwidth`: The bitwidth to be used for quantization.
- `slim_quant`: Whether to use SLiM-Quant for pruning. If set to 'False', AbsMax or OPTQ (GPTQ) will be used for quantization.
- `weight_tiled_quantization`: Whether to use weight tiled (group) quantization. We do not recommend using this option with SLiM-Quant.
- `weight_tile_size`: The size of the weight tiles to be used for weight tiled quantization.
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
- `lora_tile_size`: The size of the low-rank adapter tiles to be used for low-rank approximation. 
- `separate_lora`: Whether to keep the low-rank adapters separate from the model weights. If set to `False`, the low-rank adapters will be merged with the model weights.
- `seed`: The seed to be used for reproducibility.
- `joint_pq_mixing_factor`: The mixing factor to be used for joint pruning and quantization (JSQ).
- `calibration_dataset`: The dataset to be used for calibration.
- `pad_lora`: Whether to pad the low-rank adapters to `lora_tile_size` when not using LoRA quantizatoin.
- `scale_important_weights`: Whether to scale the important weights in quantization (similar to AWQ).
- `mask_checkpoint`: The checkpoint to use for MaskLLM pruning
- `update_weights`: Whether to use weight updates for pruning.
- `use_qp_solver`: Whether to use the QP solver for weight updates. If set to `False`, the ADAM optimizer will be used for weight updates.
- `double_precision`: Whether to use double precision for weight updates. If set to `False`, single precision will be used.
- `skip_attention`: Whether to skip pruning and quantization of attention layers.
