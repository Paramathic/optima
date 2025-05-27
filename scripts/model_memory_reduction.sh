export HF_DATASETS_TRUST_REMOTE_CODE="1"
export HF_HOME="data"
export HF_DATASETS_OFFLINE="1"
export HF_HUB_OFFLINE="1"
export HF_TOKEN="HF_TOKEN"



MODEL="meta-llama/Llama-2-13b-hf"

for QUANTIZE_LORA in '' '--quantize_lora'
do
    python speedup/model_memory_reduction.py \
        --model $MODEL \
        $QUANTIZE_LORA
done