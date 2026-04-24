## Tokenizer Design and Vocabulary Size Selection

We trained a single BPE tokenizer from scratch using only the `greek_translation` field of the TinyStories-GR training split. The tokenizer was then frozen and reused unchanged across all model runs so that vocabulary design would not become an additional confounding variable in the scaling-law analysis.

### Design Goals

The vocabulary size was not chosen arbitrarily. Instead, we selected it through a small empirical profiling sweep guided by three competing objectives:

1. **Tokenization efficiency**  
   If the vocabulary is too small, Greek words are split into too many subword units. This increases sequence length and reduces the amount of semantic content that fits into a fixed context window.

2. **Parameter efficiency in small models**  
   In the tiny-model regime, the embedding table can consume a substantial fraction of the total parameter budget. Since our smallest model is heavily parameter-constrained, we explicitly tracked the relative embedding cost of each vocabulary candidate.

3. **Vocabulary utilization**  
   TinyStories-GR is a domain-constrained corpus with relatively simple and repetitive language. If the vocabulary is too large, many learned tokens become rare, which reduces effective vocabulary utilization and allocates capacity to entries that receive few updates.

### Literature-Guided Heuristics

Our profiling was informed by literature-inspired heuristics rather than strict hard constraints. In particular:

- Following prior multilingual tokenization work, we treated **fertility** (tokens per word) as a proxy for segmentation quality in a morphologically richer language such as Greek.
- Following scaling-law work in small-model regimes, we tracked the **embedding share** to monitor whether vocabulary size was consuming too much of the parameter budget.
- We used approximate target ranges as **soft reference points**:
  - fertility roughly in the **1.35–1.60** range,
  - embedding share ideally in the **15%–25%** range for the smallest model.

These targets were used as guiding assumptions, not as strict pass/fail rules. The final decision was based on the overall empirical trade-off observed in the profiling curves and token frequency distribution.

### Tokenizer Configuration

We used a BPE tokenizer with the following configuration:

- **training text**: Greek-only text from `greek_translation`
- **normalization**: Unicode `NFKC`
- **pre-tokenization**: whitespace-based splitting
- **special tokens**: `[PAD]`, `[UNK]`, `[BOS]`, `[EOS]`

We intentionally preserved Greek orthographic information by avoiding lowercasing and accent stripping.

### Vocabulary Profiling Procedure

We profiled the candidate vocabulary sizes **4k, 8k, 12k, and 16k** on a representative subset of the training split. For each candidate, we computed:

- **Fertility**: average number of subword tokens per whitespace-delimited word
- **Embedding share**: estimated fraction of a small-model parameter budget allocated to token embeddings
- **Active vocabulary ratio**: fraction of vocabulary items that appear at least a minimum number of times in the profiling subset

The goal was to find the smallest vocabulary size after which fertility improvements started to show diminishing returns, while also monitoring embedding growth and avoiding excessive vocabulary sparsity.

### Profiling Results and Figure Interpretation

The vocabulary profiling results supported a compromise around **8,000** tokens.

The **fertility curve** decreased as vocabulary size increased, confirming that larger vocabularies reduce subword fragmentation. In our profiling sweep, fertility dropped from **1.4168** at 4k to **1.2886** at 8k, then more modestly to **1.2440** at 12k and **1.2239** at 16k. This indicates diminishing returns beyond 8k: increasing the vocabulary further still improves segmentation efficiency, but the gains become progressively smaller.

The **embedding-share curve** increased approximately linearly with vocabulary size, showing that larger vocabularies impose a direct parameter-cost penalty. This was particularly relevant for the smallest-model regime, where vocabulary growth competes with transformer capacity for a limited parameter budget. For this reason, we did not treat larger vocabularies as “free” improvements, even when they slightly reduced fertility.

The **token frequency histogram** showed a clear long-tail distribution: a relatively small number of tokens appeared very frequently, while many tokens appeared rarely. This confirmed that the corpus is lexically simple but still contains many infrequent entries. As a result, excessively large vocabularies would allocate increasing capacity to sparse tokens that receive limited training signal.

An additional useful observation is that the fertility curve exhibits an **elbow** around **8k** vocabulary items. Moving from 4k to 8k gives a meaningful reduction in fragmentation, whereas the improvements from 8k to 12k and 16k are much smaller relative to the additional vocabulary cost. This elbow pattern is one of the main reasons we treated 8k as the most balanced choice.

Taken together, the profiling figures suggest the following trade-off:
- **4k** is more compact but fragments words more aggressively,
- **12k** and **16k** reduce fertility only marginally relative to 8k,
- **8k** offers the best overall balance between tokenization efficiency, embedding cost, and vocabulary utilization.

### Final Selection

Based on this profiling process, we selected a vocabulary size of **8,000** tokens for all experiments. This choice was not derived from a single hard threshold, but from the overall empirical trade-off observed across the profiling metrics and plots. In practice, **8k** substantially reduced fragmentation relative to 4k, while avoiding the extra embedding cost and lower vocabulary utilization associated with 12k and 16k.

After selecting the final vocabulary size, we retrained the tokenizer once on the full training split and reused the saved tokenizer unchanged across the entire experimental grid.