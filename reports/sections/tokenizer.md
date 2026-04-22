## Tokenizer Design and Vocabulary Size Selection

We trained a single BPE tokenizer from scratch using only the `greek_translation` field of the TinyStories-GR training split. The tokenizer was then frozen and reused identically across all model runs in order to avoid introducing vocabulary size as an additional confounding variable in the scaling-law analysis.

### Design Goals

The vocabulary size was selected through a small empirical profiling sweep rather than fixed arbitrarily. The goal was to balance three competing factors:

1. **Tokenization efficiency**  
   If the vocabulary is too small, Greek words are fragmented into too many subword units, increasing sequence length and reducing the amount of semantic content that fits into a fixed context window.

2. **Parameter efficiency in small models**  
   In the tiny-model regime, the token embedding table can consume a non-trivial fraction of the total parameter budget. This is especially relevant for the small model configuration.

3. **Vocabulary utilization**  
   Since TinyStories-GR is a domain-constrained corpus with relatively simple language, an excessively large vocabulary risks allocating parameters to sparse or rarely updated tokens.

### Tokenizer Configuration

We used a BPE tokenizer with the following configuration:

- training text: Greek-only text from `greek_translation`
- normalization: Unicode `NFKC`
- pre-tokenization: whitespace-based splitting
- special tokens: `[PAD]`, `[UNK]`, `[BOS]`, `[EOS]`

We intentionally preserved Greek orthographic information by not applying lowercasing or accent stripping.

### Vocabulary Profiling Procedure

We evaluated several candidate vocabulary sizes on a representative subset of the training split. For each candidate, we measured:

- **Fertility**: average number of subword tokens per whitespace-delimited word
- **Embedding share**: fraction of the small model configuration’s trainable parameters allocated to token embeddings
- **Active vocabulary ratio**: fraction of vocabulary items used at least a minimum number of times in a profiling subset

We then selected the smallest vocabulary size after which fertility improvements began to show diminishing returns, while monitoring embedding cost for the small model configuration and avoiding excessive vocabulary sparsity.

### Final Selection

Based on this profiling process, we selected a vocabulary size of **8,000** tokens for all experiments. This value provided a good trade-off between lower sequence fragmentation, manageable embedding cost, and adequate vocabulary utilization for the TinyStories-GR domain.

After selecting the final vocabulary size, we retrained the tokenizer once on the full training split and reused the saved tokenizer unchanged across the entire experimental grid.
