
module load anaconda3
source activate pytorch

MODEL_PREFIX=openai-community/gpt2 #facebook/opt-
# MODEL_SIZE=-large #2.7b
STRUCTURE=unstructured
METHOD=magnitude
SPARSITY_RATIO=0.5
LORA_RANK=0
# WANDA_IN_LORA='--wanda_in_lora'
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
    $LOCAL_CHECKPOINT_DIR

    