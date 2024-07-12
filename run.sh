
#module load anaconda3 cuda/11.4.4 gcc/10.3.0 ninja
#source activate pytorch

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/compression/pruning_kernels/tensor_cores/libcusparse_lt/lib
export HF_DATASETS_TRUST_REMOTE_CODE="1"
export HF_HOME="data"
#export HF_DATASETS_OFFLINE="1"

MODEL_PREFIX=facebook/opt- #mistralai/Mistral- #meta-llama/Llama-2- #facebook/opt-
# MODEL_POSTFIX=-v0.3 #-hf

for MODEL_SIZE in 6.7b #7B #125m # 7b #1.3b #7B #6.7b
do
    for STRUCTURE in "2:4"
    do
        for METHOD in wanda #sparsegpt
        do
#           rm -rf data
            LOCAL_FILES_ONLY='--local_files_only'
            SPARSITY_RATIO=0.5
            LORA_RANK=0.1
            WANDA_IN_LORA='--wanda_in_lora'
            SHIFT_ZERO_METRICS='--shift_zero_metrics'
            EVAL_DATASET='wikitext2'
            QUANTIZE='--quantize'
            BITWIDTH=4
            QUANTIZE_INPUT='--quantize_input'
            INPUT_BITWIDTH=8
            # QUANTIZE_BEFORE_PRUNING='--quantize_before_pruning'
            MAX_BITWIDTH=4
            USE_STD_IN_QUANTIZATION='--use_std_in_quantization'
            #BIAS_CORRECTION='--bias_correction'
            EVAL_BATCH_SIZE=1
             SEPARATE_LORA='--separate_lora'
            # ACCELERATE='--accelerate'
            # RANDOMIZED_SVD='--randomized_svd'
            # LOCAL_CHECKPOINT_DIR='--local_checkpoint_dir llm_weights/flash_attn_gpt2_small_dense_lora0.pt'
#            TEST_LMHARNESS='--test_lmharness'
#            FINE_TUNE='--fine_tune'
            EVALUATE_PERPLEXITY='--evaluate_perplexity'

            CUDA_VISIBLE_DEVICES=0 python main_opt.py \
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
                $TEST_LMHARNESS \
                --output_csv_path results/perplexity.csv \
                $FINE_TUNE \
                $EVALUATE_PERPLEXITY \
                $LOCAL_FILES_ONLY \
                $QUANTIZE_INPUT \
                --input_bitwidth $INPUT_BITWIDTH

            
        done
    done
done
