
module load anaconda3 cuda/11.4.4 gcc/10.3.0 ninja
source activate pytorch

MODEL_PREFIX=facebook/opt-
MODEL_SIZE=125m
STRUCTURE=unstructured
METHOD=wanda
SPARSITY_RATIO=0.5
LORA_RANK=0.1
WANDA_IN_LORA='--wanda_in_lora'
SHIFT_ZERO_METRICS='--shift_zero_metrics'
EVAL_DATASET='wikitext2'
QUANTIZATION='--quantization'
BITWIDTH=4
QUANTIZE_BEFORE_PRUNING='--quantize_before_pruning'
MAX_BITWIDTH=8
USE_STD_IN_QUANTIZATION='--use_std_in_quantization'
BIAS_CORRECTION='--bias_correction'
EVAL_BATCH_SIZE=1
# RANDOMIZED_SVD='--randomized_svd'
# LOCAL_CHECKPOINT_DIR='--local_checkpoint_dir local_checkpoints/flash_attn_gpt2_small_dense.pt'

python main_opt.py \
    --model ${MODEL_PREFIX}${MODEL_SIZE} \
    --prune_method $METHOD \
    --sparsity_ratio $SPARSITY_RATIO \
    --sparsity_type $STRUCTURE \
    --save out/opt_$MODEL_SIZE/$STRUCTURE/$METHOD/ \
    --lora_rank $LORA_RANK \
    $WANDA_IN_LORA \
    $RANDOMIZED_SVD \
    --eval_dataset $EVAL_DATASET \
    $SHIFT_ZERO_METRICS \
    $LOCAL_CHECKPOINT_DIR \
    $QUANTIZATION \
    $QUANTIZE_BEFORE_PRUNING \
    --bitwidth $BITWIDTH \
    --max_bitwidth $MAX_BITWIDTH \
    $BIAS_CORRECTION \
    --bias_alpha 0.3 \
    --bias_correction_nsamples 16 \
    $USE_STD_IN_QUANTIZATION \
    --eval_batch_size $EVAL_BATCH_SIZE

    