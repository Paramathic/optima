
export HF_DATASETS_TRUST_REMOTE_CODE="1"
export HF_HOME="data"
#export HF_DATASETS_OFFLINE="1"
#export HF_HUB_OFFLINE="1"

HF_TOKEN="--hf_token HUGGINGFACE_ACCESS_TOKEN"

for MODEL_NAME in opt #llama2
do
    if [ $MODEL_NAME == 'llama2' ]
    then
        MODEL_PREFIX=meta-llama/Llama-2-
        MODEL_POSTFIX=-hf
        MODEL_SIZE_LIST="7b 13b"
    elif [ $MODEL_NAME == 'opt' ]
    then   
        MODEL_PREFIX=facebook/opt-
        MODEL_POSTFIX=""
        MODEL_SIZE_LIST="125m" # 350m 1.3b 2.7b" # 6.7b 13b"
    fi


    for MODEL_SIZE in $MODEL_SIZE_LIST
    do
        for STRUCTURE in 2:4 #unstructured
        do
            for METHOD in wanda #joint_pq
            do
                for LORA_RANK in 0.1
                do
                    for SLIM_LORA in '--slim_lora'
                    do
                        for NUM_CALIBRATION_SAMPLES in 128
                        do
                            for QUANTIZE_WEIGHT in '--quantize_weight'
                            do
#                                LOCAL_FILES_ONLY='--local_files_only'
                                SPARSITY_RATIO=0.5
                                SHIFT_ZERO_METRICS='--shift_zero_metrics'
                                EVAL_DATASET='wikitext2'
                                BITWIDTH=4
#                                if [ $QUANTIZE_WEIGHT == '--quantize_weight' ]; then
#                                    QUANTIZE_INPUT='--quantize_input'
##                                    TILED_INPUT_QUANTIZATION='--tiled_input_quantization'
#                                fi
                                INPUT_BITWIDTH=8
                                INPUT_GROUP_SIZE=128
                                SLIM_QUANT='--slim_quant'
                                EVAL_BATCH_SIZE=1
                                SEPARATE_LORA='--separate_lora'
#                                TEST_LMHARNESS='--test_lmharness'
                                FINE_TUNE='--fine_tune'
                                EVALUATE_PERPLEXITY='--evaluate_perplexity'
                                OPTIMIZER="adafactor"
#                                PRUNE_LORA="--prune_lora"
                                QUANTIZE_LORA="--quantize_lora"
                                LORA_TILE_SIZE=128
#                                TILED_WEIGHT_QUANTIZATION="--tiled_weight_quantization"
                                WEIGHT_TILE_SIZE=128
                                JOINT_PQ_MIXING_FACTOR=2.1

                                CUDA_VISIBLE_DEVICES=0 python main.py \
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
                                    $SLIM_QUANT \
                                    --eval_batch_size $EVAL_BATCH_SIZE \
                                    $SEPARATE_LORA \
                                    $TEST_LMHARNESS \
                                    --output_csv_path results/logs.csv \
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
                                    $HF_TOKEN \
                                    --joint_pq_mixing_factor $JOINT_PQ_MIXING_FACTOR
                            done
                        done
                    done
                done
            done
        done
    done
done
