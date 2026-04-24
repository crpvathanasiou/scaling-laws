## Experiment Grid

After validating the pipeline with a single vertical-slice training run, we executed a structured **3 × 3 experimental grid** spanning model size and training-token budget.

### Fixed Training Setup

All sweep runs used the same core training setup:

- tokenizer: frozen 8,000-token Greek BPE tokenizer
- context length: **128**
- batch size: **16**
- learning rate: **3e-4**
- weight decay: **0.01**
- evaluation frequency: every **50** steps
- evaluation batches: **20**
- seed: **42**
- device: **CUDA**

### Sweep Dimensions

The sweep varied two axes:

1. **Model size**
   - `model_1m`
   - `model_5m`
   - `model_20m`

2. **Training token budget**
   - **2M**
   - **10M**
   - **50M**

This produced a total of **9 runs**.

### Step Budget Derivation

For each run, the number of optimizer steps was derived from the token budget as:

\[
\text{tokens per step} = \text{batch size} \times (\text{context length} - 1)
\]

With batch size = 16 and context length = 128:

\[
\text{tokens per step} = 16 \times 127 = 2032
\]

The step budget for each run was then computed as:

\[
\text{max steps} = \left\lfloor \frac{\text{token budget}}{2032} \right\rfloor
\]

This ensured that each run consumed approximately the intended number of training tokens while keeping the training protocol standardized.

### Final Experiment Grid

| Run name | Model | Trainable parameters | Token budget | Actual training tokens | Max steps | Estimated FLOPs |
|---------|-------|---------------------:|-------------:|-----------------------:|----------:|----------------:|
| model_1m_tok_2000000   | model_1m  | 1,833,728  | 2,000,000  | 1,999,488   | 984   | 21,999,102,787,584 |
| model_1m_tok_10000000  | model_1m  | 1,833,728  | 10,000,000 | 9,999,472   | 4,921 | 110,017,870,749,696 |
| model_1m_tok_50000000  | model_1m  | 1,833,728  | 50,000,000 | 49,999,392  | 24,606 | 550,111,710,560,256 |
| model_5m_tok_2000000   | model_5m  | 5,451,264  | 2,000,000  | 1,999,488   | 984   | 65,398,421,716,992 |
| model_5m_tok_10000000  | model_5m  | 5,451,264  | 10,000,000 | 9,999,472   | 4,921 | 327,058,570,395,648 |
| model_5m_tok_50000000  | model_5m  | 5,451,264  | 50,000,000 | 49,999,392  | 24,606 | 1,635,359,313,788,928 |
| model_20m_tok_2000000  | model_20m | 20,866,560 | 2,000,000  | 1,999,488   | 984   | 250,334,617,927,680 |
| model_20m_tok_10000000 | model_20m | 20,866,560 | 10,000,000 | 9,999,472   | 4,921 | 1,251,927,494,737,920 |
| model_20m_tok_50000000 | model_20m | 20,866,560 | 50,000,000 | 49,999,392  | 24,606 | 6,259,891,878,789,120 |

The FLOP estimate was computed using the standard approximation:

\[
\text{FLOPs} \approx 6 \times N \times D
\]

where \(N\) is the number of trainable parameters and \(D\) is the number of training tokens actually consumed.