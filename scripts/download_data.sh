export HF_DATASETS_TRUST_REMOTE_CODE="1"
export HF_HOME="data"

python download_data.py --local_cache
python download_data.py