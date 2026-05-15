# =============================================================================
# Upload Dataset to Google Drive (run locally on your Mac)
# =============================================================================
# Run this script ONCE on your Mac to copy the dataset files to Google Drive.
# Then open the Colab notebooks and they'll find the files via Drive mount.
# =============================================================================

# ── OPTION A : Copy manually ──────────────────────────────────────────────────
# 1. Open Google Drive in Finder (install Google Drive for Desktop if not done)
# 2. Create folder:  My Drive / UPSC_SLM /
# 3. Copy these two files into it:
#    - /Users/satyamurti/Downloads/DataSets/dataset_output_final/combined/unified_pretrain.jsonl
#    - /Users/satyamurti/Downloads/DataSets/dataset_output_final/combined/unified_sft.jsonl

# ── OPTION B : Push to HuggingFace Hub (better for large files) ───────────────
# Run the cells below in Colab (or locally if you have huggingface_hub installed)

"""
from datasets import load_dataset
from huggingface_hub import HfApi

HF_TOKEN    = "your_token_here"
HF_USERNAME = "your_username_here"
REPO_ID     = f"{HF_USERNAME}/upsc-gemma2-pretrain"

# Load local JSONL
pretrain_ds = load_dataset(
    "json",
    data_files="/Users/satyamurti/Downloads/DataSets/dataset_output_final/combined/unified_pretrain.jsonl",
    split="train"
)

sft_ds = load_dataset(
    "json",
    data_files="/Users/satyamurti/Downloads/DataSets/dataset_output_final/combined/unified_sft.jsonl",
    split="train"
)

# Push to Hub (creates private repo by default)
pretrain_ds.push_to_hub(REPO_ID, config_name="pretrain", token=HF_TOKEN, private=True)
sft_ds.push_to_hub(REPO_ID, config_name="sft", token=HF_TOKEN, private=True)

print(f"Dataset pushed to https://huggingface.co/datasets/{REPO_ID}")
"""

# ── OPTION C : Check file sizes before uploading ──────────────────────────────

from pathlib import Path

files = {
    "unified_pretrain.jsonl": Path("/Users/satyamurti/Downloads/DataSets/dataset_output_final/combined/unified_pretrain.jsonl"),
    "unified_sft.jsonl":      Path("/Users/satyamurti/Downloads/DataSets/dataset_output_final/combined/unified_sft.jsonl"),
}

print("Files to upload:")
for name, path in files.items():
    if path.exists():
        size_mb = path.stat().st_size / 1024 / 1024
        lines = sum(1 for _ in open(path, encoding="utf-8"))
        print(f"  {name:<30}  {size_mb:6.1f} MB  |  {lines:,} records")
    else:
        print(f"  {name:<30}  NOT FOUND — run merge-only first:")
        print(f"  python3 scripts/pdf_extractor.py --merge-only --output dataset_output_final/")
