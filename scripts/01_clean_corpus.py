from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download # disable_progress_bars

#disable_progress_bars()

REPO = "bakrianoo/jabarti-llm-dataset"
MAX_CHUNKS = 20

HF_FILES = {
    "phase1_train": "pretrain/phase1_train-00000-of-00001.parquet",
    "phase1_eval": "pretrain/phase1_eval-00000-of-00001.parquet",
    "phase2_train": "pretrain/phase2_train-00000-of-00001.parquet",
    "phase2_eval": "pretrain/phase2_eval-00000-of-00001.parquet",
    "ft_train": "finetune/train-00000-of-00001.parquet",
    "ft_eval": "pretrain/eval-00000-of-00001.parquet",
}

TRAIN_SPLITS = {"phase1_train", "phase2_train"}

OUT_DIR = Path(__file__).parent / "output"


def download_and_filter():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for name, filename in HF_FILES.items():
        print(f"Downloading {name}...")

        local = hf_hub_download(
            repo_id=REPO,
            repo_type="dataset",
            filename=filename,
        )

        df = pd.read_parquet(local)

        print(f"{name}: {len(df):,} rows before filtering")

        if name in TRAIN_SPLITS:
            df = (
                df.sort_values(["article_id", "chunk_index"])
                .groupby("article_id", sort=False)
                .head(MAX_CHUNKS)
                .reset_index(drop=True)
            )

            print(
                f"{name}: {len(df):,} rows "
                f"after limiting to {MAX_CHUNKS} chunks/article"
            )

        output_file = OUT_DIR / f"{name}_filtered.parquet"
        
        df.to_parquet(output_file, index=False)

        print(f"Saved: {output_file}\n")


if __name__ == "__main__":
    download_and_filter()