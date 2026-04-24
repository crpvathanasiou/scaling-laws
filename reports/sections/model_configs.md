## Model Architecture and Configuration

All experiments used the same GPT-style **decoder-only causal language model** family, trained from random initialization. The architecture consisted of token embeddings, learned positional embeddings, a stack of Transformer decoder blocks with causal self-attention, and a final language-model head. We also used **tied input/output embeddings** in all model variants.

To isolate the effect of scale, we kept the following design choices fixed across the model family:

- vocabulary size: **8,000**
- context length: **128**
- MLP ratio: **4.0**
- dropout: **0.1**
- bias terms: **enabled**
- tied embeddings: **enabled**

The main scaling dimensions were therefore **depth** (`n_layer`) and **width** (`n_embd`), with the number of attention heads adjusted accordingly.

An important note is that the model names `model_1m`, `model_5m`, and `model_20m` are **nominal scale labels** rather than exact parameter counts. The actual trainable parameter counts measured from the implemented architecture are shown in Table X.

| Model name | Actual trainable parameters | Layers (`n_layer`) | Heads (`n_head`) | Embedding dim (`n_embd`) | Context length | MLP ratio | Dropout | Tied embeddings |
|-----------|-----------------------------:|-------------------:|-----------------:|-------------------------:|---------------:|----------:|--------:|----------------:|
| model_1m  | 1,833,728                    | 4                  | 4                | 128                      | 128            | 4.0       | 0.1     | Yes             |
| model_5m  | 5,451,264                    | 6                  | 8                | 224                      | 128            | 4.0       | 0.1     | Yes             |
| model_20m | 20,866,560                   | 10                 | 8                | 384                      | 128            | 4.0       | 0.1     | Yes             |

This model family was intentionally simple and consistent across runs. Rather than changing architectural components between experiments, we scaled a single decoder-only design across three sizes, which makes the observed loss trends easier to interpret in the scaling-law analysis.