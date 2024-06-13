
#module load anaconda3 cuda/11.4.4 gcc/10.3.0 ninja
#source activate pytorch

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/compression/pruning_kernels/tensor_cores/libcusparse_lt/lib

MODEL_PREFIX=meta-llama/Llama-2- #facebook/opt-
MODEL_POSTFIX=-hf

for MODEL_SIZE in 7b #6.7b
do
    for STRUCTURE in dense #"2:4"
    do
#        rm -rf data
        METHOD=sparsegpt #wanda
        SPARSITY_RATIO=0.5
        LORA_RANK=0.1
#        WANDA_IN_LORA='--wanda_in_lora'
        SHIFT_ZERO_METRICS='--shift_zero_metrics'
        EVAL_DATASET='wikitext2'
#        QUANTIZE='--quantize'
        BITWIDTH=4
        # QUANTIZE_BEFORE_PRUNING='--quantize_before_pruning'
        MAX_BITWIDTH=4
        USE_STD_IN_QUANTIZATION='--use_std_in_quantization'
        #BIAS_CORRECTION='--bias_correction'
        EVAL_BATCH_SIZE=1
        # SEPARATE_LORA='--separate_lora'
        # ACCELERATE='--accelerate'
        # RANDOMIZED_SVD='--randomized_svd'
        # LOCAL_CHECKPOINT_DIR='--local_checkpoint_dir local_checkpoints/flash_attn_gpt2_small_dense.pt'
        TEST_MMLU='--test_mmlu'

        python main_opt.py \
            --model ${MODEL_PREFIX}${MODEL_SIZE}${MODEL_POSTFIX} \
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
            $QUANTIZE \
            $QUANTIZE_BEFORE_PRUNING \
            --bitwidth $BITWIDTH \
            --max_bitwidth $MAX_BITWIDTH \
            $BIAS_CORRECTION \
            --bias_alpha 0.3 \
            --bias_correction_nsamples 16 \
            $USE_STD_IN_QUANTIZATION \
            --eval_batch_size $EVAL_BATCH_SIZE \
            $SEPARATE_LORA \
            $ACCELERATE \
            $TEST_MMLU \
            --output_csv_path results/perplexity.csv

            

    done
done
