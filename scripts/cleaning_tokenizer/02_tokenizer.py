from pathlib import Path

import pandas as pd
from tokenizers import (
    Tokenizer,
    decoders,
    models,
    pre_tokenizers,
    trainers,
)

from cleaning import build_normalizer


CURRENT_WORKING_DIR = Path(__file__).parent

CLEAN_PATHS = [
    CURRENT_WORKING_DIR / "output" / "phase1_train_filtered.parquet",
    CURRENT_WORKING_DIR / "output" / "phase2_train_cleaned.parquet",
]

OUTPUT_PATH = CURRENT_WORKING_DIR / "tokenizer.json"

VOCAB_SIZE = 32_000
MIN_FREQUENCY = 2

SPECIAL_TOKENS = [
    "[PAD]",
    "[UNK]",
    "[BOS]",
    "[EOS]",
    "[SEP]",
    "[ar]",
    "[en]",
    "[SYS]",
    "[USER]",
    "[ASST]",
]


def build_tokenizer():
    tokenizer = Tokenizer(
        models.BPE(unk_token="[UNK]")
    )

    tokenizer.normalizer = build_normalizer()

    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(
        replacement="▁",
        prepend_scheme="always",
    )

    tokenizer.decoder = decoders.Metaspace(
        replacement="▁",
        prepend_scheme="always",
    )

    return tokenizer


def corpus_iter(df, batch_size=1000):
    texts = df["text"].tolist()

    for i in range(0, len(texts), batch_size):
        yield texts[i:i + batch_size]


def train(tokenizer, df):
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQUENCY,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )

    tokenizer.train_from_iterator(
        corpus_iter(df),
        trainer=trainer,
        length=len(df),
    )

    return tokenizer


def sanity_check(tokenizer):
    print("\n" + "=" * 70)
    print("SANITY CHECK")
    print("=" * 70)

    examples = [
        "السَّلَامُ عَلَيْكُمْ وَرَحْمَةُ اللَّهِ وَبَرَكَاتُهُ",
        "مُحَمَّدٌ يَدْرُسُ عِلْمَ الحَاسُوبِ",
        "تَعَلَّمُ الآلَاتِ وَالذَّكَاءُ الاصْطِنَاعِيُّ",
        "هذا اختبار للنموذج العربي مع التشكيل",
        "Artificial intelligence is transforming modern technology.",
        "Machine learning allows computers to learn from data.",
        "The tokenizer should support both Arabic and English.",
        "This is a bilingual language model.",
    ]

    for text in examples:
        encoded = tokenizer.encode(
            text,
            add_special_tokens=False,
        )

        decoded = tokenizer.decode(encoded.ids)

        print("\nInput:")
        print(text)

        print("\nTokens:")
        print(encoded.tokens)

        print("\nIDs:")
        print(encoded.ids)

        print("\nDecoded:")
        print(decoded)

        print("-" * 70)


def main():
    print("Step 1: Load cleaned corpus")

    frames = []

    for clean_path in CLEAN_PATHS:
        frame = pd.read_parquet(clean_path)

        print(
            f"{clean_path.name}: "
            f"{len(frame):,} rows"
        )

        frames.append(frame)

    df = pd.concat(
        frames,
        ignore_index=True,
    )

    print(f"\nTotal training rows: {len(df):,}")

    print("\nStep 2: Build tokenizer")

    tokenizer = build_tokenizer()

    print("Step 3: Train BPE")

    tokenizer = train(
        tokenizer=tokenizer,
        df=df,
    )

    print("\nStep 4: Save tokenizer")

    tokenizer.save(str(OUTPUT_PATH))

    print(f"Tokenizer saved to: {OUTPUT_PATH}")

    # print("\nStep 5: Sanity check")

    # sanity_check(tokenizer)


if __name__ == "__main__":
    main()