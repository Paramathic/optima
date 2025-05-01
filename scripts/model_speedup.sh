export HF_DATASETS_TRUST_REMOTE_CODE="1"
export HF_HOME="data"
#export HF_DATASETS_OFFLINE="1"
#export HF_HUB_OFFLINE="1"
export HF_TOKEN="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"



MODEL="facebook/opt-1.3b" #'meta-llama/Llama-3.2-3B'
SEQLEN=2048
MAX_BATCHSIZE=64
SINGLE_TOKEN_GENERATOIN='--single_token_generation'
#TIME_PREFILL='--time_prefill'
INPUT_TOKEN_STEP=1024

for SETTING in quantized_slim #sparse #slim
do
    if [ $SETTING == "dense" ]; then
        COMPRESS_MODEL=''
        LORA_RANK=0
    elif [ $SETTING == "sparse" ]; then
        COMPRESS_MODEL='--compress_model'
        LORA_RANK=0
    elif [ $SETTING == "slim" ]; then
        COMPRESS_MODEL='--compress_model'
        LORA_RANK=0.1
    elif [ $SETTING == "quantized_slim" ]; then
        COMPRESS_MODEL='--compress_model'
        LORA_RANK=0.1
        QUANTIZE_LORA='--quantize_lora'
    fi

    echo "Setting: $SETTING"
    echo "**********************************************************"
    python speedup/model_speedup.py \
        --model $MODEL \
        --seqlen $SEQLEN \
        --max_batch_size $MAX_BATCHSIZE \
        $SINGLE_TOKEN_GENERATOIN \
        --input_token_step $INPUT_TOKEN_STEP \
        $TIME_PREFILL \
        $COMPRESS_MODEL \
        --lora_rank $LORA_RANK \
        $QUANTIZE_LORA
done