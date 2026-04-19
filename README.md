# Graph-Based Stylistic Compatibility and Harmony Scorer

A production-grade outfit compatibility system that fuses **deterministic
Computer Vision mathematics** with **latent semantic embeddings from CLIP**
to score outfit harmony on a 0–100 % scale.

---

## Architecture Overview

```
Raw Image ─► GrabCut Segmentation ─► Explicit CV Features ──────────────────┐
                                       (Color · Texture · Shape)             │
                                                                              ├─► SVM/MLP ─► Harmony %
Raw Image ─────────────────────────► CLIP Encoder ──► Latent Embeddings ─────┘
                                                                              │
                                     Colour Rule Engine ──► Rule Score ───────┘
                                     (complementary, analogous, triadic …)
```

The final harmony percentage is a weighted blend:

    harmony = 0.85 × ML_score  +  0.15 × colour_rule_score

---

## Pipeline Modules

| Module | File | Responsibility |
|--------|------|----------------|
| 1 | `src/preprocessing/segmenter.py` | GrabCut + GMM background removal |
| 2a | `src/features/color_harmony.py` | CIE L\*a\*b\* + K-Means dominant palette |
| 2b | `src/features/texture_analyzer.py` | Gabor filter bank texture signature |
| 2c | `src/features/shape_descriptor.py` | HOG silhouette descriptor |
| 2 | `src/features/explicit_extractor.py` | Orchestrates 2a + 2b + 2c → single vector |
| 3 | `src/features/latent_extractor.py` | Fashion-CLIP / CLIP image embeddings |
| 4a | `src/fusion/pairwise_scorer.py` | Euclidean + Cosine → pairwise feature vector |
| 4b | `src/fusion/color_rule_scorer.py` | Deterministic colour harmony rules engine |
| 4c | `src/fusion/harmony_scorer.py` | Full pipeline orchestrator; returns `HarmonyResult` |
| — | `src/models/svm_classifier.py` | sklearn Pipeline: StandardScaler → PCA → SVC |
| — | `src/models/mlp_classifier.py` | PyTorch MLP with BatchNorm, Dropout, early stopping |
| — | `src/models/model_factory.py` | Factory: `build_model(cfg)` → SVM or MLP |
| — | `src/utils/` | Dataset loader, image I/O, HDF5 cache, logging, config |
| — | `configs/config.yaml` | All hyper-parameters centralised |
| — | `scripts/` | CLI entry-points: extract, train, evaluate, infer |

---

## Dataset

**Maryland Polyvore (Cleaned)** — 20 fashion categories:

`Tops` · `Skirts` · `Pants` · `Outerwear` · `Dresses` · `Jumpsuits` · `Shoes` ·
`Bags` · `Earrings` · `Necklaces` · `Rings` · `Bracelets` · `Watches` · `Hats` ·
`Eyewear` · `Gloves` · `Legwear` · `Neckwear` · `Hairwear` · `Brooch`

See `data/README.md` for the required directory structure and JSON formats.

---

## Quick Start

```bash
# 1. Install the package in editable mode (sets up all imports correctly)
pip install -e ".[dev]"

# 2. Configure dataset paths
vim configs/config.yaml       # set paths.dataset_root / images_dir

# 3. Extract and cache all features (one-time, ~hours on first run)
python scripts/extract_features.py --split all

# 4. Train the compatibility classifier
python scripts/train.py --model mlp       # or --model svm

# 5. Evaluate on the held-out test split
python scripts/evaluate.py --split test

# 6. Score a single outfit pair interactively
python scripts/infer.py --item1 path/to/shirt.jpg --item2 path/to/pants.jpg
```
python scripts/infer.py \
  --item1 /home/yashraj/Documents/project/outfit_compatibility_final/data/polyvore/images/100219114_2.jpg \
  --item2 /home/yashraj/Documents/project/outfit_compatibility_final/data/polyvore/images/100277667_3.jpg

python scripts/infer.py \
  --item1 /home/yashraj/Documents/project/outfit_compatibility_final/data/polyvore/images/100002074_1.jpg \
  --item2 /home/yashraj/Documents/project/outfit_compatibility_final/data/polyvore/images/100002074_2.jpg


---

## CLI Reference

### `scripts/extract_features.py`

Pre-computes and caches explicit CV features (GrabCut → Color/Texture/HOG)
and CLIP latent embeddings for every item in the dataset.  Results are stored
in HDF5 files under `outputs/features/`.

```
python scripts/extract_features.py [--config PATH] [--split all|train|val|test]
                                    [--device cuda|cpu] [--dry_run]
```

### `scripts/train.py`

Loads cached features, builds pairwise feature vectors, fits a StandardScaler,
and trains the SVM or MLP classifier.  Saves the best checkpoint and scaler to
`outputs/checkpoints/`.

```
python scripts/train.py [--config PATH] [--model svm|mlp]
```

### `scripts/evaluate.py`

Evaluates the trained model on the test or validation split and prints a full
metrics report (Accuracy, ROC-AUC, AP, confusion matrix, harmony distribution).
Saves a timestamped JSON report to `outputs/results/`.

```
python scripts/evaluate.py [--config PATH] [--split test|val]
```

### `scripts/infer.py`

Scores compatibility for a single garment pair or a batch from a JSON file.

```bash
# Single pair
python scripts/infer.py --item1 shirt.jpg --item2 pants.jpg

# Batch mode
python scripts/infer.py --pairs_json my_pairs.json --output results.json
```

---

## Running Tests

```bash
# Run the full test suite
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=term-missing
```

Test modules:

| File | Coverage |
|------|----------|
| `tests/test_segmenter.py` | GrabCutSegmenter |
| `tests/test_features.py` | ColorHarmonyExtractor, TextureAnalyzer, ShapeDescriptor |
| `tests/test_color_rules.py` | ColorRuleScorer — all six harmony rules |
| `tests/test_fusion.py` | PairwiseScorer |
| `tests/test_models.py` | MLPCompatibilityClassifier, SVMCompatibilityClassifier |

---

## Output Example

```
┌─ Outfit Compatibility Scorer ──────────────────────────────┐
│  Item A : white_shirt.jpg                                   │
│  Item B : navy_chinos.jpg                                   │
│  ████████████████████████████████████░░░░░░░░░░░░░░░░░░░░ │
│  Harmony  :   73.4%  │  Verdict: Good Match                 │
│  ML prob  : 0.8012   │  Color rule: 0.4821                  │
│  CLIP cos : +0.7341  │  Eucl dist: 12.4823                  │
│  Dom.rule : analogous                                        │
│  Latency  : 284.3 ms                                        │
└─────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
outfit_compatibility/
├── configs/
│   └── config.yaml              ← all hyper-parameters
├── data/
│   ├── README.md                ← dataset setup instructions
│   └── polyvore/                ← populate with Maryland Polyvore
│       ├── images/
│       ├── train_pairs.json
│       ├── val_pairs.json
│       ├── test_pairs.json
│       └── categories.json
├── notebooks/
│   └── demo.ipynb               ← interactive walkthrough
├── outputs/
│   ├── checkpoints/             ← saved model weights
│   ├── features/                ← HDF5 feature caches
│   ├── logs/                    ← rotating log files
│   └── results/                 ← evaluation JSON reports
├── scripts/
│   ├── extract_features.py
│   ├── train.py
│   ├── evaluate.py
│   └── infer.py
├── src/
│   ├── features/
│   │   ├── color_harmony.py
│   │   ├── explicit_extractor.py
│   │   ├── latent_extractor.py
│   │   ├── shape_descriptor.py
│   │   └── texture_analyzer.py
│   ├── fusion/
│   │   ├── color_rule_scorer.py
│   │   ├── harmony_scorer.py
│   │   └── pairwise_scorer.py
│   ├── models/
│   │   ├── mlp_classifier.py
│   │   ├── model_factory.py
│   │   └── svm_classifier.py
│   ├── preprocessing/
│   │   └── segmenter.py
│   └── utils/
│       ├── config_loader.py
│       ├── dataset_loader.py
│       ├── feature_cache.py
│       ├── image_io.py
│       └── logger.py
├── tests/
│   ├── conftest.py
│   ├── test_color_rules.py
│   ├── test_features.py
│   ├── test_fusion.py
│   ├── test_models.py
│   └── test_segmenter.py
├── requirements.txt
└── setup.py
```
