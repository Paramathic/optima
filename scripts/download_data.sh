export HF_DATASETS_TRUST_REMOTE_CODE="1"
export HF_HOME=data


# Load datasets
python -m slim.dataset

# Load models and tokenizers
python -m utils.model