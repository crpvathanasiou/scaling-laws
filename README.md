# Scaling Laws for a Small Greek Language Model

This repository contains a full pipeline for:
- training a Greek BPE tokenizer from scratch
- training multiple small causal language models
- collecting scaling-law measurements
- fitting power-law trends
- producing a final report

## Tokenizer usage

The project uses a single Greek-only BPE tokenizer trained from scratch on the `greek_translation` field of `alexliap/tinystories-gr`. The tokenizer is profiled first to compare candidate vocabulary sizes, then trained once with the selected final vocabulary size and reused unchanged across all model runs.

### Run tokenizer profiling + final tokenizer training

On Windows PowerShell:

```powershell
.\scripts\01_tokenizer.ps1
```

On bash:

```bash
bash scripts/01_tokenizer.sh
```

This produces tokenizer artifacts under `artifacts/tokenizer/`, including:

- `tokenizer_profile.csv`
- `tokenizer_profile.json`
- `plots/fertility_vs_vocab_size.png`
- `plots/embedding_share_vs_vocab_size.png`
- `plots/token_frequency_histogram.png`
- `final_vocab_8000/tokenizer.json`
- `final_vocab_8000/tokenizer_config.json`
- `final_vocab_8000/special_tokens_map.json`
- `final_vocab_8000/tokenizer_preview.json`
- `final_vocab_8000/vocab.txt`

## Notes
* The tokenizer is trained only on the training split.
* No English text is used.
* The same saved tokenizer is reused across the full scaling-law sweep.

