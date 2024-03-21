
module load anaconda3
source activate pytorch

MODEL_SIZE=13b
STRUCTURE=unstructured
METHOD=wanda
SPARSITY_RATIO=0.5
LORA_RANK=0.05
WANDA_IN_LORA='--wanda_in_lora'
# RANDOMIZED_SVD='--randomized_svd'

python main_opt.py \
    --model facebook/opt-$MODEL_SIZE \
    --prune_method $METHOD \
    --sparsity_ratio $SPARSITY_RATIO \
    --sparsity_type $STRUCTURE \
    --save out/opt_$MODEL_SIZE/$STRUCTURE/$METHOD/ \
    --lora_rank $LORA_RANK \
    $WANDA_IN_LORA \
    $RANDOMIZED_SVD

    