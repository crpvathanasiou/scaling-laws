# Α Model Architecture Overview

We implemented a **GPT-style decoder-only Transformer language model** from scratch in PyTorch. The model is designed for **causal language modeling**, meaning that at each position it predicts the **next token** given only the tokens to its left.

The architecture follows the standard high-level GPT pattern:

1. **Input token ids** are mapped to dense vector representations through a learned **token embedding** layer.
2. A learned **positional embedding** is added so that the model can distinguish token order inside the sequence.
3. The resulting sequence representation is passed through a **stack of Transformer blocks**.
4. A final **LayerNorm** is applied to the hidden states.
5. A final **language modeling head** projects hidden states to vocabulary logits, from which next-token probabilities are computed.

## Input Representation

The model operates on tokenized sequences of shape **(B, T)**, where:

* **B** is the batch size
* **T** is the sequence length

Each token id is first converted into a learned embedding vector through:

* `self.token_embedding`

In parallel, each sequence position is represented by a learned positional embedding through:

* `self.position_embedding`

These two embeddings are summed:

```text
x = token_embedding + positional_embedding
```

This sum forms the initial input representation for the Transformer stack.

## Α1 Transformer Stack

The core of the model is a stack of **`n_layer` Transformer blocks**:

* `self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])`

For example, in `configs/model_small.yaml`, the setting:

```yaml
n_layer: 4
```

means that the input sequence passes through **4 Transformer blocks sequentially**.

Each **Block** is a **pre-LayerNorm Transformer block** and contains:

### 1. Causal Self-Attention

Implemented in:

* `CausalSelfAttention`

This module allows each token to attend only to:

* itself
* previous tokens

but **not future tokens**.

This behavior is enforced by:

```python
F.scaled_dot_product_attention(..., is_causal=True)
```

which makes the model suitable for **autoregressive next-token prediction**.

### 2. Feed-Forward Network (MLP)

Implemented in:

* `MLP`

This is the standard Transformer feed-forward sublayer:

```text
Linear → GELU → Linear → Dropout
```

It transforms the hidden representation at each position independently.

### 3. Residual Connections and LayerNorm

Implemented in:

* `Block`

Each block applies:

```text
x = x + attention(LayerNorm(x))
x = x + mlp(LayerNorm(x))
```

So every block contains:

* one LayerNorm before attention
* one residual connection around attention
* one LayerNorm before the MLP
* one residual connection around the MLP

This is the standard **pre-LN Transformer design**, which improves training stability.

## Α2 Final Layers

After all Transformer blocks, the model applies a final normalization layer:

* `self.ln_f`

Then the hidden states are projected to vocabulary logits using:

* `self.lm_head`

This produces output of shape:

* **(B, T, vocab_size)**

so that for every position in the sequence, the model predicts a probability distribution over the vocabulary.

If `tie_embeddings=True`, the output projection shares weights with the input token embedding table:

```python
self.lm_head.weight = self.token_embedding.weight
```

This is a common language-modeling design choice that reduces parameter count and keeps input/output token representations aligned.

## Α3 Forward Pass Summary

The full forward path is:

```text
token ids
→ token embeddings
+ positional embeddings
→ Transformer block 1
→ Transformer block 2
→ ...
→ Transformer block N
→ final LayerNorm
→ language modeling head
→ vocabulary logits
```

For the small model configuration with `n_layer: 4`, this becomes:

```text
token ids
→ token embeddings + positional embeddings
→ block 1
→ block 2
→ block 3
→ block 4
→ final LayerNorm
→ lm head
```

## Α4 Training Objective

The model is trained with **token-level cross-entropy loss**.
When target token ids are provided, the forward method computes:

```python
F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
```

This means the model is optimized to predict the correct next token at each position in the sequence.

---

# Β GPT Design Patterns Followed

To ensure that the implemented model follows the **GPT architecture pattern** rather than a generic Transformer, we adopted the following design choices.

## Β1. Decoder-only Transformer design

The implemented model is **decoder-only**, not encoder-decoder and not encoder-only.

This means the model is designed for **autoregressive language modeling**: it processes a sequence from left to right and predicts the next token at each position.

This design is reflected in the fact that the model consists of:

* token embeddings
* positional embeddings
* a stack of Transformer decoder blocks
* a final language modeling head

without any encoder stack or cross-attention mechanism.

## Β2. Causal self-attention

A defining GPT pattern is that attention must be **causal**.

In practice, this means that each token can attend only to:

* itself
* previous tokens

but not to future tokens.

This property is implemented in the attention layer through:

```python id="3wsg9h"
F.scaled_dot_product_attention(..., is_causal=True)
```

This is one of the most important reasons why the model is GPT-style: it enforces the left-to-right prediction constraint required for next-token generation.

## Β3. Autoregressive next-token prediction objective

GPT models are trained to predict the **next token** in a sequence.

This pattern is followed in the implementation because the model outputs logits over the vocabulary for every sequence position, and training uses token-level cross-entropy loss against shifted targets.

So the model is not trained for:

* masked token reconstruction
* sequence classification
* encoder-decoder generation

It is trained specifically for **autoregressive language modeling**.

## Β4. Repeated Transformer block structure

Another GPT pattern is the use of a repeated stack of identical Transformer blocks.

In this implementation, the model depth is controlled by:

* `n_layer`

and the blocks are created as:

```python id="h8r0nh"
self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
```

Each block follows the standard GPT/Transformer decoder pattern:

* self-attention sublayer
* feed-forward sublayer
* residual connections
* normalization

This repeated-block design is central to GPT-style scaling.

## Β5. Pre-LayerNorm residual architecture

The implemented block uses a **pre-LN** formulation:

```python id="es12wg"
x = x + self.attn(self.ln_1(x))
x = x + self.mlp(self.ln_2(x))
```

This means normalization is applied **before** each sublayer.

This is a widely used modern GPT-style training pattern because it improves optimization stability, especially as models become deeper.

## Β6. Learned token embeddings and learned positional embeddings

The model uses:

* learned token embeddings
* learned positional embeddings

rather than sinusoidal positional encodings or handcrafted input features.

This is consistent with the GPT family design, where both token identity and token position are learned directly from data.

## Β7. Language modeling head over the vocabulary

The final output layer maps hidden states to vocabulary logits through a linear projection:

* `self.lm_head`

This is the standard GPT output pattern:
for each sequence position, the model produces a distribution over the full vocabulary and selects the most likely next token.

## Β8. Optional weight tying between input and output embeddings

The implementation supports:

* `tie_embeddings: true`

When enabled, the output projection weights are shared with the input embedding table:

```python id="h6i3eq"
self.lm_head.weight = self.token_embedding.weight
```

This is a common GPT-style optimization that:

* reduces parameter count
* improves parameter efficiency
* keeps input and output token representations aligned

## Β9. Random initialization and training from scratch

The model does not import pretrained GPT weights.

Instead, it initializes weights from scratch using standard Gaussian initialization for linear and embedding layers.

This is important for the assignment because the requirement is to train models **from scratch**, not to fine-tune a pretrained model.

---

## Summary

The model follows the GPT architecture pattern because it combines:

* a **decoder-only Transformer structure**
* **causal self-attention**
* **autoregressive next-token prediction**
* a **stack of repeated Transformer blocks**
* **pre-LayerNorm residual design**
* **learned embeddings**
* a **vocabulary projection head**
* optional **embedding weight tying**

Together, these design decisions make the implementation a **GPT-style causal language model**, even though it is implemented directly in PyTorch rather than imported from a pretrained GPT library.

---


# C Architecture-to-Code Mapping

This section maps each major architectural component of the implemented GPT-style model to the corresponding code elements in `src/scaling_laws/models/gpt.py`.

## C1. Model configuration

The overall model shape is defined by the `GPTConfig` dataclass.

Relevant code:

* `GPTConfig`
* `GPTConfig.from_dict(...)`

This component is responsible for translating the YAML configuration into model hyperparameters such as:

* vocabulary size
* context length
* embedding dimension
* number of layers
* number of attention heads
* dropout
* weight tying

In particular, `from_dict(...)` maps the project configuration keys:

* `n_embd`
* `n_layer`
* `n_head`

to the internal model fields:

* `d_model`
* `n_layers`
* `n_heads`

## C2. Token embedding layer

The token embedding layer is implemented as:

* `self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)`

This layer converts discrete token ids into dense continuous vectors of dimension `d_model`.

## C3. Positional embedding layer

The positional embedding layer is implemented as:

* `self.position_embedding = nn.Embedding(config.context_length, config.d_model)`

This layer provides a learned representation of token positions in the sequence.

## C4. Input representation construction

The input representation is formed in the `forward(...)` method of `GPTLanguageModel` through:

* `tok_emb = self.token_embedding(idx)`
* `pos_emb = self.position_embedding(positions)`
* `x = self.dropout(tok_emb + pos_emb)`

This is the point where token identity and token position are combined before entering the Transformer stack.

## C5. Transformer block stack

The repeated Transformer blocks are implemented as:

* `self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])`

This means the model depth is determined directly by `config.n_layers`.

For example:

* if `n_layer = 4`, then the model contains 4 stacked Transformer blocks

## C6. Transformer block structure

A single Transformer block is implemented in:

* `class Block(nn.Module)`

Inside `Block`, the two main sublayers are:

* `self.attn = CausalSelfAttention(config)`
* `self.mlp = MLP(config)`

and the two normalization layers are:

* `self.ln_1 = nn.LayerNorm(config.d_model)`
* `self.ln_2 = nn.LayerNorm(config.d_model)`

The forward pass of one block is:

```python id="7wpk46"
x = x + self.attn(self.ln_1(x))
x = x + self.mlp(self.ln_2(x))
```

This code implements:

* LayerNorm before attention
* residual connection around attention
* LayerNorm before the MLP
* residual connection around the MLP

## C7. Causal self-attention

The self-attention mechanism is implemented in:

* `class CausalSelfAttention(nn.Module)`

Key architectural elements in this class are:

### Query / Key / Value projection

* `self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)`

This creates Q, K, and V in one projection for efficiency.

### Multi-head split

Inside `forward(...)`, the tensor is reshaped into multiple heads:

* `q = q.view(...).transpose(...)`
* `k = k.view(...).transpose(...)`
* `v = v.view(...).transpose(...)`

### Causal masking

The GPT-style causal constraint is implemented by:

```python id="fug1e2"
F.scaled_dot_product_attention(
    q,
    k,
    v,
    attn_mask=None,
    dropout_p=self.dropout if self.training else 0.0,
    is_causal=True,
)
```

The argument:

* `is_causal=True`

is the key implementation detail that makes this attention layer autoregressive.

### Output projection

After merging heads back together, attention output is projected through:

* `self.out_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)`

## C8. Feed-forward network (MLP)

The feed-forward network inside each block is implemented in:

* `class MLP(nn.Module)`

Its layers are:

* `self.fc = nn.Linear(config.d_model, hidden_dim, bias=config.bias)`
* `self.proj = nn.Linear(hidden_dim, config.d_model, bias=config.bias)`

and the forward path is:

```python id="8ms3al"
x = self.fc(x)
x = F.gelu(x)
x = self.proj(x)
x = self.dropout(x)
```

This corresponds to the standard Transformer MLP pattern:

* linear expansion
* GELU activation
* projection back to model dimension
* dropout

## C9. Final layer normalization

After the full Transformer stack, the model applies a final normalization layer:

* `self.ln_f = nn.LayerNorm(config.d_model)`

This is used in the main forward pass as:

* `x = self.ln_f(x)`

## C10. Language modeling head

The final projection to vocabulary logits is implemented as:

* `self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)`

This produces logits of shape:

* `(B, T, vocab_size)`

so that each sequence position predicts the next token distribution over the vocabulary.

## C11. Weight tying

Optional weight tying is implemented through:

```python id="ljfm6g"
if config.tie_embeddings:
    self.lm_head.weight = self.token_embedding.weight
```

This connects:

* input token embeddings
* output vocabulary projection

using the same weight matrix.

## C12. Weight initialization

Model initialization is implemented in:

* `def _init_weights(self, module):`

This function initializes:

* linear layers with Gaussian weights
* embedding layers with Gaussian weights
* biases to zero

The initialization is applied through:

* `self.apply(self._init_weights)`

## C13. Forward pass

The main forward computation is implemented in:

* `def forward(self, idx, targets=None):`

This method:

1. checks that sequence length does not exceed the configured context length
2. builds position indices
3. looks up token and positional embeddings
4. sums them
5. passes the result through all Transformer blocks
6. applies final layer normalization
7. projects to vocabulary logits
8. optionally computes cross-entropy loss if `targets` are provided

## C14. Loss computation

The training loss is implemented directly in the forward method through:

```python id="jlwmh0"
loss = F.cross_entropy(
    logits.reshape(-1, logits.size(-1)),
    targets.reshape(-1),
)
```

This corresponds to standard token-level next-token prediction loss.

## C15. Model construction helper

The convenience entrypoint for constructing the model is:

* `build_model(model_cfg: dict, vocab_size: int)`

This function:

1. builds a `GPTConfig`
2. instantiates `GPTLanguageModel`
3. returns the model object

This is the main interface used by the training code.

---

# Short Mapping Summary

* **model hyperparameters** → `GPTConfig`
* **config-file translation** → `GPTConfig.from_dict(...)`
* **token embeddings** → `self.token_embedding`
* **positional embeddings** → `self.position_embedding`
* **Transformer stack** → `self.blocks`
* **one Transformer block** → `Block`
* **causal attention** → `CausalSelfAttention`
* **feed-forward network** → `MLP`
* **final normalization** → `self.ln_f`
* **vocabulary projection head** → `self.lm_head`
* **weight tying** → `self.lm_head.weight = self.token_embedding.weight`
* **loss computation** → `F.cross_entropy(...)`
* **model builder** → `build_model(...)`

---



