## Scaling Law Fits

After completing the 9-run experimental grid, we analyzed how validation loss scales with model size, training tokens, and total compute. We fitted simple one-dimensional power laws of the form:

\[
L(x) = a \cdot x^b
\]

where \(L\) is the final validation loss and \(x\) is one of:
- number of parameters \(N\),
- number of training tokens \(D\),
- estimated FLOPs.

All fits were performed in log-log space.

### Loss vs Parameters

The fitted relation for validation loss as a function of parameter count was:

\[
L(N) \approx 8.1233 \cdot N^{-0.0634}
\]

with:

- exponent: **-0.0634**
- \(R^2\) in log-space: **0.0610**

This fit is weak. Although larger models generally performed better at fixed token budgets, a single global 1D fit of loss versus parameter count did not explain much of the variance in the full 9-run grid. This is expected, because the parameter axis is confounded by simultaneous variation in training-token budget.

### Loss vs Training Tokens

The fitted relation for validation loss as a function of training tokens was:

\[
L(D) \approx 60.6623 \cdot D^{-0.1861}
\]

with:

- exponent: **-0.1861**
- \(R^2\) in log-space: **0.9176**

This was the strongest of the three 1D fits. It indicates a clear and consistent scaling trend with data: as the number of training tokens increases, validation loss decreases in a predictable power-law-like manner across the experiment grid.

### Loss vs FLOPs

The fitted relation for validation loss as a function of estimated compute was:

\[
L(C) \approx 345.2765 \cdot C^{-0.1414}
\]

with:

- exponent: **-0.1414**
- \(R^2\) in log-space: **0.8334**

This fit is also strong, showing that the final validation loss improves systematically as total compute increases.

### Interpretation

Taken together, the fitted exponents show that the cleanest scaling signal in our experiments comes from **training tokens** and **compute**, while a simple global 1D fit with **parameter count alone** is much less informative.

This does **not** mean that model size is unimportant. In the raw experiment grid, larger models usually outperform smaller ones at the same token budget. However, because both model size and token budget vary simultaneously across runs, the parameter-only fit is not sufficient to isolate that effect.

A practical interpretation of the fitted trends is:

- **data scaling** showed the clearest and most stable empirical relationship,
- **compute scaling** also showed a strong monotonic trend,
- **parameter scaling**, when viewed in isolation as a single 1D curve across all runs, was comparatively weak.

### Relation to the Figures

- **Figure X** (`loss_vs_params.png`) shows that the parameter-only fit is weak and that the runs do not collapse cleanly to a single curve when token budget varies.
- **Figure Y** (`loss_vs_tokens.png`) shows the clearest power-law behavior, with loss decreasing consistently as training tokens increase.
- **Figure Z** (`loss_vs_flops.png`) shows a similarly strong compute trend, though slightly noisier than the token-based fit.

### Scope of the Fitting

These fits are intentionally simple empirical 1D fits. They are useful for visualizing the main trends in the sweep and for comparing them with the scaling-law literature. However, they are not a full joint fit of the form:

\[
L(N, D) = E + A N^{-\alpha} + B D^{-\beta}
\]

Therefore, the exponents reported here should be interpreted as descriptive summaries of the observed trends in this small-scale experimental setup rather than as definitive estimates of universal scaling constants.