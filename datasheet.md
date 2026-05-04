# Datasheet for CSI-Text

This document follows the structure recommended in *Datasheets for Datasets* (Gebru et al.) and summarizes the **CSI-Text** multimodal dataset derived from public WiFi CSI corpora and synthetic or LLM-augmented natural-language descriptions.

---

## Motivation

### For what purpose was the dataset created?

The dataset was created to **enable research on language-grounded ambient perception from commodity WiFi channel state information (CSI)**. By pairing normalized CSI amplitude traces with natural-language descriptions at multiple semantic levels, CSI-Text supports training and evaluation of models that align radio-frequency sensing with text—supporting applications such as interpretable monitoring, open-vocabulary sensing, and instruction-driven analysis without device-specific engineering for each new task description.

### Who created the dataset and on behalf of which entity?

The dataset was created by the **CSI-Text Multimodal Alignment** project authors (see repository maintainers). It is intended as an academic research resource.

### What (other) tasks could the dataset be used for?

Beyond multimodal alignment, derived tasks include **zero-shot or few-shot WiFi sensing**, **retrieval** (CSI-to-text and text-to-CSI), **scene or attribute question answering**, and **analysis of representation geometry** (e.g., embedding visualization and linear probing).

---

## Composition

### What does the dataset consist of?

CSI-Text consists of **CSI–text pairs** built from **three source datasets**:

| Source | Role in CSI-Text |
|--------|------------------|
| **CSI-Bench** | Diverse indoor sensing tasks (e.g., fall and breath-related events, motion sources, localization, HAR, user identity, proximity). |
| **XRF55** | Fine-grained human activity and related tasks (e.g., action recognition, user, scene). |
| **SignFi** | Sign-language and environment-oriented tasks (e.g., sign recognition, user, environment). |

Each logical sample associates a **normalized CSI amplitude tensor** (see Preprocessing) with one or more **text strings**. The pairing structure is **many-to-many**: a single CSI segment may link to multiple text descriptions (e.g., template and paraphrase variants, or composite captions), and the same text string may correspond to multiple CSI samples of the same semantic class.

**Hierarchical text levels** include:

- **Level 1**: Class-centric semantic captions (e.g., per-class templates describing the activity or scene).
- **Level 2**: Composite captions that combine metadata (e.g., user, activity, proximity) where available.

### Are there recommended splits?

The reference codebase uses **train / validation / test** splits with a **70% / 15% / 15%** ratio (configurable per source dataset). Users should cite the split definition used in their experiments.

### Personal information

The dataset is built from **public CSI benchmarks** and **synthetic or paraphrased text**. It is not intended to contain raw biometric identifiers; nonetheless, **user identity** may appear as **categorical labels** in some tasks. Users must comply with the licenses of the **original** CSI sources and with applicable privacy norms when redistributing or combining data.

---

## Collection Process

### How was the data collected?

- **CSI signals** were obtained from the **CSI-Bench**, **XRF55**, and **SignFi** distributions, following each source’s documented acquisition setup (commodity WiFi hardware, controlled or semi-controlled environments, and task-specific labels as defined by those works).
- **Text descriptions** were produced through a **template-based pipeline** grounded in task and class definitions, with optional **LLM paraphrasing** (e.g., multiple diverse phrasings per template) to increase lexical diversity while preserving semantics. A **quality filter** enforces basic length and validity constraints on generated strings.

### Over what timeframe was the data collected?

CSI segments reflect the **collection periods and protocols of the respective source papers and releases**, not a single new physical collection campaign for CSI-Text. Refer to CSI-Bench, XRF55, and SignFi for hardware and environment details.

### How was the data associated with its labels?

Labels are inherited from each source dataset’s published task definitions (e.g., class IDs for HAR, sign glosses, room indices). Text entries are **aligned to those labels** when building the many-to-many registry.

---

## Preprocessing

### What preprocessing was done?

The unified pipeline (see `data/preprocessing/unified_preprocess.py`) applies:

1. **Amplitude extraction and tensor shaping**: Raw CSI is converted to a canonical **channel–subcarrier–time** layout where applicable (handles multiple raw layouts per source).
2. **Subcarrier and temporal standardization**: **Interpolation** maps inputs to a fixed **subcarrier count** and **time length** (reference configuration: **64** subcarriers, **500** time steps).
3. **Per-sample amplitude normalization**: Traces are normalized to stabilize scale across devices and environments.
4. **Unified on-disk layout**: Each task directory ships with `metadata/sample_metadata.csv`, `splits/{train,val,test}_id.json`, a `pair_registry.json` describing the many-to-many CSI–text mapping, and per-sample HDF5 files referenced by `file_path` in the CSV. Encoder-specific text embedding caches (`text_embeddings_<encoder>.h5`) are generated on demand by the training pipeline. The HDF5 dataset key inside each per-sample file follows the source convention (`CSI_amps` for CSI-Bench, dynamic per-sample for XRF55, `amplitude` for SignFi); a consolidated layout with key `csi_data` is also produced when users opt into `scripts/preprocess_to_consolidated.py` for faster I/O.

Preprocessing choices affect comparability; experiments should fix these hyperparameters when reporting results.

---

## Uses

### What tasks has the dataset been used for?

The intended uses include:

- **Multimodal alignment**: Train models that map CSI and text into a shared embedding space (contrastive, redundancy-reduction, masked prediction, and related objectives).
- **Zero-shot WiFi sensing**: Classify or rank CSI using **text prototypes** without task-specific fine-tuning.
- **Ambient QA**: Answer multiple-choice or attribute questions by scoring CSI embeddings against candidate answer strings.

### Is there anything that dataset consumers should be aware of?

CSI is **sensitive to environment and hardware**; **domain shift** across rooms, devices, and layouts remains a fundamental limitation. Text is **template- and model-generated** to varying degrees; it does not replace expert annotations for safety-critical deployment. Results should be interpreted as **research benchmarks**, not clinical or legal evidence.

---

## Distribution

### How will the dataset be distributed?

The dataset and metadata are distributed through:

- **Hugging Face Hub** (primary tabular/registry files and documentation),
- **Kaggle** (mirror for accessibility),
- **GitHub** (code, configs, and generation utilities).

### License

The **CSI-Text** release is shared under **CC-BY-SA-4.0** (see `croissant.json` and the repository **README**). **Upstream CSI data** remain subject to their **original licenses**; redistribution must respect those terms.

### Maintenance

**Who maintains the dataset?**

The **paper authors / repository maintainers** listed in the GitHub repository are responsible for versioning, documentation updates, and issue triage for the CSI-Text release.

**How can owners be contacted?**

Use the contact channels provided on the **GitHub repository** (issues and maintainer email if published).

**Will the dataset be updated?**

Future revisions may add splits, fix preprocessing bugs, or extend text generation; version tags and changelogs will accompany substantive updates.


### Source CSI datasets

```bibtex

@inproceedings{zhu2025csibench,
  title  = {{CSI-Bench}: A Large-Scale In-the-Wild Dataset for Multi-Task WiFi Sensing},
  author = {Zhu, Guozhen and Hu, Yuqian and Gao, Weihang and Wang, Wei-Hsiang and Wang, Beibei and Liu, K. J. Ray},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year   = {2025}
}

@article{wang2024xrf55,
  title   = {{XRF55}: A Radio Frequency Dataset for Human Indoor Action Analysis},
  author  = {Wang, Fei and Lv, Yizhe and Zhu, Mengdie and Ding, Han and Han, Jinsong},
  journal = {Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies (IMWUT)},
  volume  = {8},
  number  = {1},
  pages   = {1--34},
  year    = {2024},
  doi     = {10.1145/3643529}
}

@article{ma2018signfi,
  title   = {{SignFi}: Sign Language Recognition Using {WiFi}},
  author  = {Ma, Yongsen and Zhou, Gang and Wang, Shuangquan and Zhao, Hongyang and Jung, Woosub},
  journal = {Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies (IMWUT)},
  volume  = {2},
  number  = {1},
  pages   = {1--21},
  year    = {2018},
  doi     = {10.1145/3191755}
}

```

### Datasheet methodology

```bibtex
@article{gebru2021datasheets,
  title   = {Datasheets for Datasets},
  author  = {Gebru, Timnit and Morgenstern, Jamie and Vecchione, Briana and Vaughan, Jennifer Wortman and Wallach, Hanna and Daum{\'e}, III, Hal and Crawford, Kate},
  journal = {Communications of the ACM},
  volume  = {64},
  number  = {12},
  pages   = {86--92},
  year    = {2021},
  doi     = {10.1145/3458723}
}
```

### Note on upstream attribution

CSI-Bench, XRF55, and SignFi remain under their original licenses. Whenever CSI portions of those datasets are used, the **original** dataset papers (above) **must** be cited in addition to CSI-Text.
