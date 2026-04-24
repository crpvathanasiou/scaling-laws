## Compute-Optimal Frontier

To complement the scaling-law fits, we computed an **empirical compute frontier** from the 9-run experimental grid. The frontier was defined directly from the observed runs: after sorting runs by increasing estimated FLOPs, a run was placed on the frontier if it achieved a strictly lower validation loss than all cheaper runs.

This gives a practical notion of **compute-optimality** within our sweep: for a given compute budget, which \((N, D)\) allocation achieved the best observed validation loss?

### Empirical Frontier Runs

The following runs appeared on the empirical frontier:

| Run name | Parameters | Training tokens | FLOPs | Validation loss |
|---------|-----------:|----------------:|------:|----------------:|
| model_1m_tok_2000000   | 1,833,728  | 1,999,488   | 21,999,102,787,584   | 4.3757 |
| model_5m_tok_2000000   | 5,451,264  | 1,999,488   | 65,398,421,716,992   | 4.0800 |
| model_1m_tok_10000000  | 1,833,728  | 9,999,472   | 110,017,870,749,696  | 3.1986 |
| model_5m_tok_10000000  | 5,451,264  | 9,999,472   | 327,058,570,395,648  | 2.8659 |
| model_1m_tok_50000000  | 1,833,728  | 49,999,392  | 550,111,710,560,256  | 2.5505 |
| model_5m_tok_50000000  | 5,451,264  | 49,999,392  | 1,635,359,313,788,928 | 2.2496 |
| model_20m_tok_50000000 | 20,866,560 | 49,999,392  | 6,259,891,878,789,120 | 2.0714 |

### Interpretation

The frontier shows that **larger models were not always compute-optimal at lower token budgets**. In particular, the 20M-parameter model only appeared on the frontier at the **highest token budget (50M tokens)**. The runs:

- `model_20m_tok_2000000`
- `model_20m_tok_10000000`

did not appear on the frontier, because cheaper alternatives achieved lower validation loss before those runs became competitive.

This is an important result. It suggests that in this small-scale setup, allocating compute to a **moderate model trained on more data** was often better than allocating the same general budget to a much larger but less well-trained model.

Another useful observation is that the frontier alternates between the 1M and 5M models over a broad range of compute budgets. This means that the best allocation of compute was not “always maximize model size,” but rather depended on the interaction between:

- model size,
- token budget,
- and total compute.

### Main Takeaway

The empirical frontier supports a Chinchilla-style interpretation of the sweep: **compute-optimal training depends on the balance between model size and data, not on model size alone**. In our runs, the best low-to-mid budget choices were often smaller or medium-sized models trained on more tokens, while the largest model became competitive only when paired with the largest training-token budget.

### Relation to the Figure

**Figure X** shows the empirical compute frontier in FLOPs-loss space. The frontier traces the best observed validation loss reachable at increasing compute budgets and highlights which runs are dominated and which are compute-optimal within the experimental grid.
