# Ablation Study Results

All ablations evaluated on CSI-Bench HAR with clip_infonce + resnet18_1d.

## Text Encoder Ablation (Priority 5B)

| Text Encoder | Zero-Shot Acc | Weighted F1 |
|:-------------|:-------------:|:-----------:|
| Sentence-Transformer | 90.2 | 90.5 |
| CLIP ViT-B/32 (ref) | ~98.9 | — |

## Text Level Ablation (Priority 3A)

| Text Level | Zero-Shot Acc | Weighted F1 |
|:-----------|:-------------:|:-----------:|
| Level 1 + Level 2 | 97.7 | 97.8 |

## Many-to-Many Paraphrase Ablation (Priority 3B)

| Method | Paraphrases | Zero-Shot Acc | Weighted F1 |
|:-------|:-----------:|:-------------:|:-----------:|
| CLIP InfoNCE | 10 | 98.3 | 98.3 |
| SigLIP | 10 | 98.3 | 98.4 |

## Embedding Dimension Ablation (Priority 5C)

| Dimension | Zero-Shot Acc | Weighted F1 |
|:---------:|:-------------:|:-----------:|
| 512 | 97.9 | 97.9 |
| 256 (default) | 95.5 | 95.6 |
