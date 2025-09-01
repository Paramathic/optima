export HF_DATASETS_TRUST_REMOTE_CODE="1"
export HF_HOME="data"

export HF_DATASETS_OFFLINE="1"
export HF_HUB_OFFLINE="1"

export CUDA_VISIBLE_DEVICES="0"

HF_TOKEN_ARG="--hf_token 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"
export HF_TOKEN="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"

export TRITON_CACHE_DIR="/tmp"

export WANDB_MODE="offline"

for MODEL_NAME in gemma2 # llama3.2 #opt #llama2 #llama3.1
do
    if [ $MODEL_NAME == 'llama2' ]
    then
        MODEL_PREFIX=meta-llama/Llama-2-
        MODEL_POSTFIX=-hf
        MODEL_SIZE_LIST="7b" # 13b"
    elif [ $MODEL_NAME == 'opt' ]
    then   
        MODEL_PREFIX=facebook/opt-
        MODEL_POSTFIX=""
        MODEL_SIZE_LIST="125m" #30b"
    elif [ $MODEL_NAME == 'llama3.2' ]
    then
        MODEL_PREFIX=meta-llama/Llama-3.2-
        MODEL_SIZE_LIST="3B"
        MODEL_POSTFIX=""
    elif [ $MODEL_NAME == 'llama3.1' ]
    then
        MODEL_PREFIX=meta-llama/Llama-3.1-
        MODEL_SIZE_LIST="8B"
        MODEL_POSTFIX=""
    elif [ $MODEL_NAME == 'llama3' ]
    then
        MODEL_PREFIX=meta-llama/Meta-Llama-3-
        MODEL_SIZE_LIST="8B"
        MODEL_POSTFIX=""
    elif [ $MODEL_NAME == 'gemma3' ]
    then
        MODEL_PREFIX=google/gemma-3-
        MODEL_SIZE_LIST="1b"
        MODEL_POSTFIX="-pt"
    elif [ $MODEL_NAME == 'gemma2' ]
    then
        MODEL_PREFIX=google/gemma-2-
        MODEL_SIZE_LIST="2b"
        MODEL_POSTFIX=""
    fi

    for MODEL_SIZE in $MODEL_SIZE_LIST
    do
        for STRUCTURE in 2:4 #unstructured
        do
            for METHOD in wanda #maskllm sparsegpt joint_pq
            do
                for LORA_RANK in 0 #0.1
                do
                    for SLIM_LORA in '--slim_lora' #''
                    do
                        for NUM_CALIBRATION_SAMPLES in 128
                        do
                            for QUANTIZE_WEIGHT in ''
                            do
                                for TILED_WEIGHT_QUANTIZATION in '--tiled_weight_quantization'
                                do
                                    LOCAL_FILES_ONLY='--local_files_only'
                                    SPARSITY_RATIO=0.5
                                    SHIFT_ZERO_METRICS='--shift_zero_metrics'
                                    EVAL_DATASET='wikitext2'
                                    BITWIDTH=4
                                    QUANT_TYPE="--quant_type symmetric"
                                    INPUT_GROUP_SIZE=256
                                    # SLIM_QUANT='--slim_quant'
                                    EVAL_BATCH_SIZE=1
                                    SEPARATE_LORA='--separate_lora'
                                    # TEST_LMHARNESS='--test_lmharness'
                                    # FINE_TUNE='--fine_tune'
                                    EVALUATE_PERPLEXITY='--evaluate_perplexity'
                                    OPTIMIZER="adafactor"
    #                                PRUNE_LORA="--prune_lora"
                                    QUANTIZE_LORA="--quantize_lora"
                                    LORA_TILE_SIZE=128
                                    WEIGHT_TILE_SIZE=128
                                    JOINT_PQ_MIXING_FACTOR=2.1
                                    CALIBRATION_DATASET="c4"
                                    # QUANTIZE_INPUT="--quantize_input"
                                    INPUT_BITWIDTH=8
                                    INPUT_GROUP_SIZE=-1
                                    PAD_LORA='--pad_lora'
#                                    SCALE_IMPORTANT_WEIGHTS='--scale_important_weights'
                                    MASKLLM_CHECKPOINT="--maskllm_checkpoint Vinnnf/LLaMA-3-8B-MaskLLM-C4"
                                    # WANDB="--use_wandb"
                                    WANDB_PROJECT="weight_update"
                                    SAVE_CHECKPOINT_PATH="--save_checkpoint_path checkpoints/${MODEL_NAME}_${MODEL_SIZE}_${METHOD}_${STRUCTURE}_lr${LORA_RANK}_sparsity${SPARSITY_RATIO}"
                                    QP_SOLVER="--use_qp_solver"
                                    UPDATE_WEIGHTS="--update_weights"
                                    # DOUBLE_PRECISION="--double_precision"

                                    python main.py \
                                        --model ${MODEL_PREFIX}${MODEL_SIZE}${MODEL_POSTFIX} \
                                        --prune_method $METHOD \
                                        --sparsity_ratio $SPARSITY_RATIO \
                                        --sparsity_type $STRUCTURE \
                                        --lora_rank $LORA_RANK \
                                        $SLIM_LORA \
                                        --eval_dataset $EVAL_DATASET \
                                        $SHIFT_ZERO_METRICS \
                                        $QUANTIZE_WEIGHT \
                                        --bitwidth $BITWIDTH \
                                        $QUANT_TYPE \
                                        $SLIM_QUANT \
                                        --eval_batch_size $EVAL_BATCH_SIZE \
                                        $SEPARATE_LORA \
                                        $TEST_LMHARNESS \
                                        --output_csv_path results/qp.csv \
                                        $FINE_TUNE \
                                        $EVALUATE_PERPLEXITY \
                                        $LOCAL_FILES_ONLY \
                                        $QUANTIZE_INPUT \
                                        --input_bitwidth $INPUT_BITWIDTH \
                                        --input_group_size $INPUT_GROUP_SIZE \
                                        --nsample $NUM_CALIBRATION_SAMPLES \
                                        --optimizer $OPTIMIZER \
                                        $TILED_INPUT_QUANTIZATION \
                                        $PRUNE_LORA \
                                        $QUANTIZE_LORA \
                                        --lora_tile_size $LORA_TILE_SIZE \
                                        $TILED_WEIGHT_QUANTIZATION \
                                        --weight_tile_size $WEIGHT_TILE_SIZE \
                                        $HF_TOKEN_ARG \
                                        --joint_pq_mixing_factor $JOINT_PQ_MIXING_FACTOR \
                                        --calibration_dataset $CALIBRATION_DATASET \
                                        $PAD_LORA \
                                        $SCALE_IMPORTANT_WEIGHTS \
                                        $MASKLLM_CHECKPOINT \
                                        $WANDB \
                                        $SAVE_CHECKPOINT_PATH \
                                        --wandb_project $WANDB_PROJECT \
                                        $UPDATE_WEIGHTS \
                                        $QP_SOLVER \
                                        $DOUBLE_PRECISION
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done
