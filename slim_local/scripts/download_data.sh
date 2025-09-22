export HF_DATASETS_TRUST_REMOTE_CODE="1"
export HF_HOME=data


# Load datasets
python -m slim.data

# Load models and tokenizers
python -m utils.model