## Limitations

This study has several important limitations that should be considered when interpreting the reported scaling trends.

### Small-Scale Experimental Regime

Although the experimental design was inspired by large-scale scaling-law studies, all runs were conducted in a much smaller compute regime. The models, token budgets, and total FLOP budgets were intentionally scaled down so that the full sweep could be executed on a consumer GPU. As a result, the fitted exponents should be interpreted as **small-scale empirical trends** rather than precise estimates of large-scale language-model scaling laws.

### Limited Grid Size

The final sweep contains **9 runs** in total, corresponding to a 3 × 3 grid over model size and token budget. This is sufficient to reveal clear monotonic trends, but still too small for highly stable estimation of universal scaling exponents. In particular, sparse coverage of the \((N, D)\) space makes the analysis more sensitive to local design choices.

### Single Dataset and Single Domain

All experiments were run on `alexliap/tinystories-gr`, a Greek translation of TinyStories. This corpus is valuable because it provides a large Greek-only training signal, but it is also domain-constrained: the language is simple, repetitive, and targeted toward children's stories. Therefore, the measured scaling behavior may not generalize directly to broader Greek corpora or more diverse language-modeling settings.

### Single Seed per Run

Each experiment was run with **one random seed only**. This means the results do not capture run-to-run variance due to initialization, data ordering, or optimization noise. Repeating each run with multiple seeds would provide more robust estimates and confidence intervals for both the measured losses and the fitted scaling exponents.

### Simplified Fitting Procedure

The scaling analysis used simple **1D log-log power-law fits** for:
- loss vs parameters,
- loss vs training tokens,
- loss vs FLOPs.

These fits are useful descriptive summaries, but they are not equivalent to a full joint scaling-law fit of the form \(L(N, D)\). This limitation is especially visible in the weak parameter-only fit, where model size and token budget vary simultaneously and cannot be fully disentangled by a single 1D regression.

### Fixed Context Length and Shared Training Protocol

All runs used the same tokenizer, the same context length (**128**), and the same general optimizer/training setup. This was the correct choice for experimental control, but it also means that the reported scaling behavior is conditional on this specific setup. Different context lengths, optimization schedules, or regularization choices could shift the observed exponents and frontier.

### Empirical Frontier, Not a Theoretical Optimum

The compute frontier reported in this work is an **empirical frontier over the observed runs**, not a closed-form optimum derived from a continuous fitted objective. It identifies the best configurations within the tested grid, but it does not guarantee that those configurations are globally optimal outside the sampled experiment space.

### Practical Compute Constraints

The experiments were executed under practical hardware constraints on a single NVIDIA RTX 3060 (12GB VRAM). Approximate sweep runtimes were:

- **small model sweep:** ~10 minutes
- **medium model sweep:** ~25–30 minutes
- **large model sweep:** ~50–60 minutes

These constraints influenced the final grid design, the number of runs, and the choice to keep the study compact and fully reproducible.

### What We Would Improve With More Compute

Given more compute, the most important next improvements would be:

1. run multiple seeds per configuration,
2. increase the number of model sizes and token budgets,
3. fit a joint \(L(N, D)\) scaling law rather than only 1D projections,
4. evaluate on broader Greek-language corpora beyond TinyStories-GR,
5. explore larger context lengths and longer training schedules.