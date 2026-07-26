# Many-to-Many Retrieval Metrics — SageMaker Runbook

Post-hoc re-scoring of the existing 42 checkpoints (6 methods x 7 tasks) with
retrieval metrics that respect the dataset's many-to-many structure — mAP,
nDCG, R-Precision and MRR over the **full caption gallery** — plus a
template-overlap audit and a saturation/headroom report.

**No retraining.** One CSI encode pass per checkpoint, same cost profile as the
earlier post-hoc eval sweeps.

## Why this exists

`RetrievalEvaluator` (`evaluation/retrieval.py`) is class-prototype retrieval,
not many-to-many retrieval. It ranks `C` mean-pooled class prototypes — 5 for
HAR, 4 for proximity, 2 for fall detection — so its R@1 is arithmetically the
same number as zero-shot accuracy, and R@5/R@10 are skipped whenever
`k <= len(class_ids)`. The per-caption embeddings it computes are never used.

`evaluation/retrieval_m2m.py` ranks against every unique caption in the pair
registry and scores each query against **all** of its valid matches, which is
what the registry's `sample_to_texts` / `text_to_samples` graph actually
defines.

## What gets reported

Two relevance rules are scored from a single ranking:

| Level | Positives | HAR example |
|---|---|---|
| `strict` | the sample's own registry edges | 9 of 221 captions, chance mAP 0.062 |
| `class_level` | any caption of the sample's class | ~62 of 221, chance mAP 0.30 |

`strict` is the instance-level view: the model must rank the sample's own
Level-2 composite above composites describing the same activity by a different
person at a different distance. `class_level` is the class-level ranking metric.
For tasks without Level-2 composites the two coincide, and the output records
that with `levels_identical`.

Per direction and level: `mAP`, `mAP@{10,100}`, `nDCG@{10,100}`, `R-Precision`,
`MRR`, `recall/precision/hit_rate@{1,5,10}`, macro-by-class and worst-class mAP,
an exact analytic chance baseline, and `normalized_mAP = (mAP - chance)/(1 - chance)`.
Graded nDCG (own caption > same-class caption > unrelated) is reported once per
direction since it does not depend on a threshold.

---

## Setup

```bash
cd /home/sagemaker-user/CSI-Text-Multimodal-Alignment
export PYTHONPATH="$(pwd)"
ROOT=/home/sagemaker-user/Multimodal_CSI_Text/Dataset
MODEL=/home/sagemaker-user/Multimodal_CSI_Text/Model
```

## Step 0 — Validate the metric implementations

Runs before touching any checkpoint. Checks the metrics against a brute-force
reference and against `sklearn`, checks the closed-form chance mAP against a
Monte-Carlo simulation, checks the near-duplicate grouping, and confirms that
random embeddings land on the analytic chance mAP for a real registry.

```bash
python scripts/validate_m2m_metrics.py \
  --registry $ROOT/csibench/multitask/har/pair_registry.json
```

Exits non-zero if any check fails. Takes a few seconds.

## Step 1 — Template-overlap audit (CPU, no checkpoint)

```bash
python scripts/analyze_template_overlap.py \
  --data_root $ROOT \
  --output_dir paper_results/template_overlap
```

Per task: registry scale, word-level Jaccard within versus between classes with
the collision rate at J > 0.5, CLIP-space 1-NN same-class rate, a text-only
logistic probe on the class label, and a census of captions the frozen text
encoder cannot separate at cosine > 0.95.

Add `--skip_embeddings` for the lexical statistics alone if the text encoder is
not available.

## Step 2 — Dry-run the sweep

```bash
DRY_RUN=1 bash scripts/run_m2m_retrieval_evals.sh
```

Prints the command for each of the 42 cells and warns about any missing
checkpoint or registry. Fix path problems here, not halfway through the sweep.

## Step 2.5 — Preflight: check every checkpoint, registry and split actually loads

```bash
python scripts/preflight_m2m.py \
  --data_root $ROOT \
  --model_root $MODEL \
  --json_out paper_results/m2m_retrieval/preflight.json
```

The dry run only checks that paths exist. This one loads them. Per cell it
confirms the checkpoint loads and its state dict is accepted with
`strict=True`, the registry parses, the test split builds, one real batch flows
through the CSI encoder, and the output directory is writable.

It also checks the one thing a successful-looking run can still get wrong:
whether the sample ids coming out of the dataloader are present in the pair
registry. If they are not, every query is silently dropped and the resulting
mAP is meaningless, so this is reported as a hard failure rather than a
warning.

Exits non-zero if any cell is blocked, and prints a list of the blocked cells
with the reason. Add `--quick` to skip the batch forward pass (no H5 reads, no
inference) when you only want the path and state-dict checks.

## Step 3 — Run the sweep

```bash
bash scripts/run_m2m_retrieval_evals.sh
```

Writes to `$MODEL/results/m2m_retrieval/<dataset>/<task>/<method>_resnet18_1d/`,
a separate tree, so nothing existing is overwritten. Pass `--merge-in-place` to
merge into the original results directories instead (it backs up each
`metrics.json` first).

Narrow the sweep while iterating:

```bash
ONLY_TASKS="csibench:har" ONLY_METHODS="clip_infonce" \
  bash scripts/run_m2m_retrieval_evals.sh
```

## Step 4 — Aggregate into paper tables

```bash
python scripts/aggregate_m2m_retrieval.py \
  --results_root $MODEL/results/m2m_retrieval \
  --output_dir paper_results/m2m_retrieval
```

Builds the 7-task x 6-method matrices for mAP (both relevance levels),
normalized mAP, macro mAP, worst-class mAP and nDCG@10, plus the Rank-1 versus
mAP contrast, the near-duplicate control, and the chance baselines. Logs a
warning listing any missing task/method cells.

## Step 5 — Saturation and headroom report

```bash
python scripts/analyze_benchmark_difficulty.py \
  --output_dir paper_results/m2m_retrieval \
  --overlap_json paper_results/template_overlap/overlap_stats.json
```

Joins the sweep with the reference numbers from `paper_results/RESULTS_ORGANIZED.md`
and the template-overlap audit. Both inputs are optional; missing sections are
skipped rather than failing.

---

## Outputs

| Artifact | Location |
|---|---|
| Preflight report | `paper_results/m2m_retrieval/preflight.json` |
| Per-checkpoint metrics | `$MODEL/results/m2m_retrieval/<dataset>/<task>/<method>_resnet18_1d/metrics.json` |
| Per-checkpoint summary | same directory, `summary.md` and `tables.tex` |
| Template overlap | `paper_results/template_overlap/{summary.md,tables.tex,overlap_stats.json}` |
| Aggregated results | `paper_results/m2m_retrieval/{summary.md,tables.tex,m2m_retrieval.json}` |
| Headroom report | `paper_results/m2m_retrieval/benchmark_difficulty.{md,tex,json}` |
| Separability figure | `paper_results/m2m_retrieval/difficulty_vs_separability.png` |

## Interpretation

- **mAP versus hit@1.** `hit@1` is the many-to-many analogue of Rank-1: did any
  valid caption land first? It saturates long before mAP does. The size of
  `hit@1 - mAP` is how much of the headline retrieval number comes from only
  ever being asked for one correct caption.
- **Normalized mAP.** Gallery size and positive count differ per task, so raw
  mAP is not comparable across tasks. Normalized mAP rescales onto
  [chance -> 1] and is the number to quote when arguing about headroom.
- **Macro and worst-class mAP.** If macro mAP sits well below micro mAP, a
  frequent class is carrying the score. Worst-class mAP is where the remaining
  difficulty is concentrated.
- **Near-duplicate control.** Merging shrinks the gallery *and* the positive
  count, so chance mAP moves too. Compare the normalized columns, not the raw
  mAP columns, across the collapse.
- **Text-side separability.** A 1-NN same-class rate near 1.0 means the caption
  alone already determines the class, so a high CSI-to-Text R@1 is partly a
  property of the caption vocabulary rather than of the alignment.

## Notes and gotchas

- The protocol needs `sample_ids` in each batch to link a CSI sample back to its
  registry edges. `CSITextDataset.collate_fn` already emits them; a custom
  collate that drops them will raise `KeyError`.
- Test samples absent from the registry are dropped with a warning and counted
  in `gallery.num_samples_dropped`. A large count means the registry and the
  split files disagree.
- The near-duplicate control is skipped above a 20,000-caption gallery
  (`MAX_DEDUP_GALLERY`), where the pairwise cosine matrix dominates the runtime.
- Duplicate grouping runs on the **frozen** text encoder's output, before the
  learned projection, so the grouping is identical across checkpoints and the
  control is comparable between methods.
- Reference numbers in `analyze_benchmark_difficulty.py` (`TASK_REFERENCE`) are
  transcribed from `paper_results/RESULTS_ORGANIZED.md` Table 1a. Update them if
  those tables are regenerated.

## Optional extension

`pair_registry_freeform_ww.json` holds ~192k unique HAR captions over the same
55,684 samples, i.e. near-zero template overlap. Scoring the existing HAR
checkpoints against that gallery would be the most direct answer to the
template-overlap question and is still post-hoc, but the gallery is far past
`MAX_DEDUP_GALLERY` and the `[N, G]` similarity matrix needs chunking, so it is
not wired into the sweep. The overlap audit can already be pointed at it:

```bash
python scripts/analyze_template_overlap.py \
  --registry har_freeform=$ROOT/csibench/multitask/har/pair_registry_freeform.json \
  --skip_embeddings \
  --output_dir paper_results/template_overlap_freeform
```
