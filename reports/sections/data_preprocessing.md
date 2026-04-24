## Dataset Preparation and Tokenized Training Data

We used the Hugging Face dataset `alexliap/tinystories-gr` and retained only the `greek_translation` field as the training corpus. No English text was used at any stage of tokenizer training, data preparation, or model training.

### Preprocessing Policy

The preprocessing logic was intentionally kept simple and consistent with the tokenizer pipeline. Each example was processed as follows:

1. keep only entries that are valid strings,
2. apply `strip()` to remove leading and trailing whitespace,
3. discard empty entries.

This ensured that the text seen during data preparation matched the assumptions used when training the tokenizer, avoiding train-time inconsistencies between tokenization and downstream model inputs.

### Train/Validation Split

The dataset was loaded from the Hugging Face `train` split and then shuffled with a fixed random seed (`seed = 42`). A **row-level** split was used to create the final train and validation sets:

- **training fraction:** 99%
- **validation fraction:** 1%

Importantly, the split was performed at the level of complete stories rather than at the token level, so that validation examples remained disjoint from training examples.

The final split sizes were:

- **training texts:** 2,120,004
- **validation texts:** 21,414

### Frozen Tokenizer Reuse

Data preparation reused the frozen tokenizer selected in the tokenizer stage:

- `artifacts/tokenizer/final_vocab_8000/tokenizer.json`

No new tokenizer was trained during this stage. This ensured that all experiments used the exact same tokenization scheme and vocabulary of **8,000** tokens.

### Tokenization and Sequence Construction

Each story was encoded with the frozen tokenizer into token IDs. Because the tokenizer includes post-processing with boundary markers, each encoded story automatically contains sequence boundary tokens such as `[BOS]` and `[EOS]`.

After tokenization, all token IDs within each split were concatenated into one continuous stream:

- one stream for the training split,
- one stream for the validation split.

These streams were then segmented into fixed-length chunks of:

- **context length = 128 tokens**

This chunking strategy produced dense fixed-shape arrays for causal language modeling. Any final incomplete remainder shorter than 128 tokens was dropped.

### Final Tokenized Dataset Statistics

The final tokenized outputs were:

- **training tokens:** 482,438,836
- **validation tokens:** 4,879,001
- **training chunks:** 3,769,053
- **validation chunks:** 38,117

The resulting tensor shapes were:

- **train:** `(3,769,053, 128)`
- **validation:** `(38,117, 128)`

The number of discarded remainder tokens was negligible:

- **training dropped tokens:** 52
- **validation dropped tokens:** 25

### Saved Outputs

The prepared data stage produced the following runtime artifacts under `artifacts/data/`:

- `train_chunks.npy`
- `val_chunks.npy`
- `data_prep_metadata.json`
- `sample_batch_preview.json`

These files were then used directly by the training stage. In particular, the `.npy` chunk arrays served as the fixed tokenized inputs for all subsequent model runs in the experimental grid.
