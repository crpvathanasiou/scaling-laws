import os
import json
import argparse
import random
from collections import Counter

import matplotlib.pyplot as plt
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers, processors


# Special tokens used consistently across all tokenizer runs.
SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]


def set_seed(seed: int):
    """Set Python's random seed for reproducibility."""
    random.seed(seed)


def clean_text(text):
    """
    Basic text cleaning:
    - keep only string values
    - strip surrounding whitespace
    - drop empty strings
    """
    if not isinstance(text, str):
        return None
    text = text.strip()
    if not text:
        return None
    return text


def load_and_split_dataset(
    dataset_name: str,
    text_column: str,
    val_size: float,
    seed: int,
):
    """
    Load the Hugging Face dataset, keep only valid text rows,
    shuffle them, and split them into train/validation sets.
    """
    ds = load_dataset(dataset_name, split="train")
    ds = ds.shuffle(seed=seed)

    filtered_texts = []
    for x in ds[text_column]:
        t = clean_text(x)
        if t is not None:
            filtered_texts.append(t)

    n_total = len(filtered_texts)
    n_val = max(1, int(n_total * val_size))
    n_train = n_total - n_val

    train_texts = filtered_texts[:n_train]
    val_texts = filtered_texts[n_train:]

    return train_texts, val_texts


def yield_texts(texts):
    """Yield texts one by one for tokenizer training from an iterator."""
    for t in texts:
        yield t


def build_tokenizer(vocab_size: int, min_frequency: int):
    """
    Build a BPE tokenizer from scratch.

    Configuration:
    - BPE model with [UNK] token
    - NFKC Unicode normalization
    - whitespace pre-tokenization
    """
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )

    return tokenizer, trainer


def add_special_token_postprocessor(tokenizer: Tokenizer):
    """
    Add a post-processor so that every encoded sequence receives:
    [BOS] ... [EOS]

    This is useful for causal language modeling experiments.
    """
    bos_id = tokenizer.token_to_id("[BOS]")
    eos_id = tokenizer.token_to_id("[EOS]")

    tokenizer.post_processor = processors.TemplateProcessing(
        single="[BOS] $A [EOS]",
        pair="[BOS] $A [EOS] $B:1 [EOS]:1",
        special_tokens=[
            ("[BOS]", bos_id),
            ("[EOS]", eos_id),
        ],
    )
    return tokenizer


def compute_fertility(tokenizer: Tokenizer, texts, sample_size: int):
    """
    Compute fertility = average number of subword tokens per whitespace word.

    Lower fertility usually means less word fragmentation.
    """
    total_words = 0
    total_tokens = 0

    for text in texts[:sample_size]:
        words = text.split()
        enc = tokenizer.encode(text)
        total_words += len(words)
        total_tokens += len(enc.ids)

    if total_words == 0:
        return float("nan")

    return total_tokens / total_words


def compute_active_vocab_ratio(
    tokenizer: Tokenizer,
    texts,
    sample_size: int,
    min_occurrences: int = 10,
):
    """
    Compute the fraction of vocabulary items that appear at least
    `min_occurrences` times in the sampled texts.

    This is a simple proxy for vocabulary utilization.
    """
    token_counts = Counter()

    for text in texts[:sample_size]:
        enc = tokenizer.encode(text)
        token_counts.update(enc.ids)

    vocab_size = tokenizer.get_vocab_size()
    active_vocab = sum(1 for _, c in token_counts.items() if c >= min_occurrences)

    return active_vocab / max(vocab_size, 1)


def estimate_embedding_share(
    vocab_size: int,
    d_model: int,
    total_params: int,
    tied_embeddings: bool = True,
):
    """
    Estimate the fraction of model parameters used by token embeddings.

    If embeddings are tied, embedding parameters are approximately:
        vocab_size * d_model

    If untied, we approximate:
        2 * vocab_size * d_model
    """
    emb_params = vocab_size * d_model
    if not tied_embeddings:
        emb_params *= 2
    return emb_params / total_params


def profile_candidate(
    vocab_size: int,
    texts_for_training,
    texts_for_metrics,
    min_frequency: int,
    fertility_sample_size: int,
    d_model: int,
    total_params: int,
    tied_embeddings: bool,
):
    """
    Train a temporary tokenizer for a given vocabulary size and compute
    profiling metrics used to choose the final vocabulary:
    - fertility
    - active vocabulary ratio
    - embedding share
    """
    tokenizer, trainer = build_tokenizer(vocab_size=vocab_size, min_frequency=min_frequency)
    tokenizer.train_from_iterator(yield_texts(texts_for_training), trainer=trainer)
    tokenizer = add_special_token_postprocessor(tokenizer)

    fertility = compute_fertility(tokenizer, texts_for_metrics, fertility_sample_size)
    active_vocab_ratio = compute_active_vocab_ratio(tokenizer, texts_for_metrics, fertility_sample_size)
    embedding_share = estimate_embedding_share(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=d_model,
        total_params=total_params,
        tied_embeddings=tied_embeddings,
    )

    return {
        "vocab_size": tokenizer.get_vocab_size(),
        "fertility": fertility,
        "embedding_share": embedding_share,
        "active_vocab_ratio": active_vocab_ratio,
    }


def save_json(obj, path: str):
    """Save a Python object as pretty-printed JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_profile_csv(rows, path: str):
    """Save tokenizer profiling results as a CSV file."""
    header = ["vocab_size", "fertility", "embedding_share", "active_vocab_ratio"]
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            vals = [
                str(row["vocab_size"]),
                f'{row["fertility"]:.6f}',
                f'{row["embedding_share"]:.6f}',
                f'{row["active_vocab_ratio"]:.6f}',
            ]
            f.write(",".join(vals) + "\n")


def ensure_dir(path: str):
    """Create a directory if it does not already exist."""
    os.makedirs(path, exist_ok=True)


def save_special_tokens_map(tokenizer: Tokenizer, output_dir: str):
    """Save a simple special tokens map for downstream reuse."""
    special_tokens_map = {
        "pad_token": "[PAD]",
        "unk_token": "[UNK]",
        "bos_token": "[BOS]",
        "eos_token": "[EOS]",
        "pad_token_id": tokenizer.token_to_id("[PAD]"),
        "unk_token_id": tokenizer.token_to_id("[UNK]"),
        "bos_token_id": tokenizer.token_to_id("[BOS]"),
        "eos_token_id": tokenizer.token_to_id("[EOS]"),
    }
    save_json(special_tokens_map, os.path.join(output_dir, "special_tokens_map.json"))


def count_token_frequencies(tokenizer: Tokenizer, texts, sample_size: int):
    """
    Count token frequencies on a sample of texts.

    Special tokens are excluded so the histogram reflects corpus token usage
    rather than BOS/EOS template insertion.
    """
    special_ids = {
        tokenizer.token_to_id("[PAD]"),
        tokenizer.token_to_id("[UNK]"),
        tokenizer.token_to_id("[BOS]"),
        tokenizer.token_to_id("[EOS]"),
    }

    counts = Counter()
    for text in texts[:sample_size]:
        enc = tokenizer.encode(text)
        for token_id in enc.ids:
            if token_id not in special_ids:
                counts[token_id] += 1
    return counts


def plot_token_frequency_histogram(token_counts: Counter, tokenizer: Tokenizer, output_path: str):
    """
    Plot a histogram of token frequencies across the sampled corpus.
    """
    if not token_counts:
        return

    ensure_dir(os.path.dirname(output_path))

    frequencies = list(token_counts.values())

    plt.figure(figsize=(8, 5))
    plt.hist(frequencies, bins=50)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Token frequency")
    plt.ylabel("Number of tokens")
    plt.title("Token Frequency Histogram")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_profile_metric(rows, metric_key: str, ylabel: str, title: str, output_path: str):
    """
    Plot a profiling metric against vocabulary size.
    """
    if not rows:
        return

    ensure_dir(os.path.dirname(output_path))

    rows_sorted = sorted(rows, key=lambda x: x["vocab_size"])
    x = [row["vocab_size"] for row in rows_sorted]
    y = [row[metric_key] for row in rows_sorted]

    plt.figure(figsize=(7, 4.5))
    plt.plot(x, y, marker="o")
    plt.xlabel("Vocabulary size")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def train_final_tokenizer(
    train_texts,
    vocab_size: int,
    min_frequency: int,
    output_dir: str,
    metadata: dict,
):
    """
    Train the final tokenizer on the full training split and save:
    - tokenizer.json
    - tokenizer_config.json
    - vocab.txt
    - special_tokens_map.json
    """
    tokenizer, trainer = build_tokenizer(vocab_size=vocab_size, min_frequency=min_frequency)
    tokenizer.train_from_iterator(yield_texts(train_texts), trainer=trainer)
    tokenizer = add_special_token_postprocessor(tokenizer)

    ensure_dir(output_dir)

    tokenizer_json_path = os.path.join(output_dir, "tokenizer.json")
    tokenizer.save(tokenizer_json_path)

    config = {
        "vocab_size": tokenizer.get_vocab_size(),
        "special_tokens": SPECIAL_TOKENS,
        "normalization": "NFKC",
        "pre_tokenizer": "Whitespace",
        "model": "BPE",
        "min_frequency": min_frequency,
        "post_processor": {
            "type": "TemplateProcessing",
            "single": "[BOS] $A [EOS]",
        },
        **metadata,
    }

    save_json(config, os.path.join(output_dir, "tokenizer_config.json"))
    save_special_tokens_map(tokenizer, output_dir)

    # Save the learned vocabulary in token-id order for inspection/debugging.
    vocab = tokenizer.get_vocab()
    vocab_sorted = sorted(vocab.items(), key=lambda x: x[1])
    with open(os.path.join(output_dir, "vocab.txt"), "w", encoding="utf-8") as f:
        for token, idx in vocab_sorted:
            f.write(f"{idx}\t{token}\n")

    return tokenizer


def main():
    """
    Main entry point.

    Modes:
    - profile: train temporary tokenizers for multiple vocab sizes and save diagnostics
    - final: train and save the final tokenizer only
    - both: run profiling first, then train the final tokenizer
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="alexliap/tinystories-gr")
    parser.add_argument("--text_column", type=str, default="greek_translation")
    parser.add_argument("--val_size", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--mode", type=str, choices=["profile", "final", "both"], default="both")
    parser.add_argument("--profile_vocab_sizes", type=int, nargs="+", default=[4000, 8000, 12000, 16000])
    parser.add_argument("--final_vocab_size", type=int, default=8000)
    parser.add_argument("--min_frequency", type=int, default=2)

    parser.add_argument("--profile_train_subset", type=int, default=50000)
    parser.add_argument("--profile_eval_subset", type=int, default=5000)

    parser.add_argument("--smallest_model_d_model", type=int, default=128)
    parser.add_argument("--smallest_model_total_params", type=int, default=1000000)
    parser.add_argument("--tied_embeddings", action="store_true")

    parser.add_argument("--token_histogram_sample_size", type=int, default=5000)

    parser.add_argument("--artifacts_dir", type=str, default="artifacts/tokenizer")
    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dir(args.artifacts_dir)

    # Load and split the dataset once. The tokenizer is trained only on train_texts.
    train_texts, val_texts = load_and_split_dataset(
        dataset_name=args.dataset_name,
        text_column=args.text_column,
        val_size=args.val_size,
        seed=args.seed,
    )

    # Save split metadata for reproducibility.
    split_meta = {
        "dataset_name": args.dataset_name,
        "text_column": args.text_column,
        "seed": args.seed,
        "val_size": args.val_size,
        "n_train_texts": len(train_texts),
        "n_val_texts": len(val_texts),
    }
    save_json(split_meta, os.path.join(args.artifacts_dir, "data_split_metadata.json"))

    # Use a smaller subset for quick vocabulary profiling.
    profile_train_texts = train_texts[:args.profile_train_subset]
    profile_eval_texts = train_texts[:args.profile_eval_subset]

    profile_rows = []
    plots_dir = os.path.join(args.artifacts_dir, "plots")
    ensure_dir(plots_dir)

    if args.mode in ["profile", "both"]:
        print("\nProfiling candidate vocabulary sizes...\n")
        print(f"{'vocab_size':>10} | {'fertility':>10} | {'embedding_share':>16} | {'active_vocab_ratio':>18}")
        print("-" * 64)

        for vocab_size in args.profile_vocab_sizes:
            metrics = profile_candidate(
                vocab_size=vocab_size,
                texts_for_training=profile_train_texts,
                texts_for_metrics=profile_eval_texts,
                min_frequency=args.min_frequency,
                fertility_sample_size=args.profile_eval_subset,
                d_model=args.smallest_model_d_model,
                total_params=args.smallest_model_total_params,
                tied_embeddings=args.tied_embeddings,
            )
            profile_rows.append(metrics)

            print(
                f"{metrics['vocab_size']:10d} | "
                f"{metrics['fertility']:10.4f} | "
                f"{metrics['embedding_share']:16.4f} | "
                f"{metrics['active_vocab_ratio']:18.4f}"
            )

        # Save profiling outputs for later analysis and report writing.
        save_profile_csv(profile_rows, os.path.join(args.artifacts_dir, "tokenizer_profile.csv"))
        save_json(profile_rows, os.path.join(args.artifacts_dir, "tokenizer_profile.json"))

        plot_profile_metric(
            rows=profile_rows,
            metric_key="fertility",
            ylabel="Fertility",
            title="Fertility vs Vocabulary Size",
            output_path=os.path.join(plots_dir, "fertility_vs_vocab_size.png"),
        )

        plot_profile_metric(
            rows=profile_rows,
            metric_key="embedding_share",
            ylabel="Embedding share",
            title="Embedding Share vs Vocabulary Size",
            output_path=os.path.join(plots_dir, "embedding_share_vs_vocab_size.png"),
        )

    if args.mode in ["final", "both"]:
        print("\nTraining final tokenizer...\n")
        final_dir = os.path.join(args.artifacts_dir, f"final_vocab_{args.final_vocab_size}")

        metadata = {
            "trained_on": "full_training_split_only",
            "selected_vocab_size": args.final_vocab_size,
        }

        tokenizer = train_final_tokenizer(
            train_texts=train_texts,
            vocab_size=args.final_vocab_size,
            min_frequency=args.min_frequency,
            output_dir=final_dir,
            metadata=metadata,
        )

        # Save one encoded example for quick sanity checking.
        sample_text = train_texts[0]
        enc = tokenizer.encode(sample_text)

        preview = {
            "sample_text": sample_text,
            "sample_ids": enc.ids[:64],
            "sample_tokens": enc.tokens[:64],
            "final_vocab_size_actual": tokenizer.get_vocab_size(),
        }
        save_json(preview, os.path.join(final_dir, "tokenizer_preview.json"))

        token_counts = count_token_frequencies(
            tokenizer=tokenizer,
            texts=train_texts,
            sample_size=args.token_histogram_sample_size,
        )
        plot_token_frequency_histogram(
            token_counts=token_counts,
            tokenizer=tokenizer,
            output_path=os.path.join(plots_dir, "token_frequency_histogram.png"),
        )

        print(f"Final tokenizer saved to: {final_dir}")
        print(f"Final vocab size: {tokenizer.get_vocab_size()}")

    print("\nDone.")


if __name__ == "__main__":
    main()
