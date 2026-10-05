# Water-Pipe Vibration — Tap-State Classification (ML Pipeline)

> End-to-end machine-learning pipeline that classifies the **open/closed state of 4
> taps** on a water pipe from raw **vibration-sensor** measurements — from parsing
> hundreds of raw sensor CSVs through to a PyTorch 1D-CNN.

**Tech stack:** Python · pandas · NumPy · scikit-learn · PyTorch

### Highlights
- Built a full ML workflow: **data ingestion → aggregation → multi-sensor fusion → modelling → evaluation**.
- Fused two independent vibration sensors by time-interval overlap, handling missing readings with `np.nan` imputation.
- Engineered temporal features (rolling max-abs / std / mean over a 10-second window) that lifted accuracy dramatically.
- Compared a shallow MLP, a Decision Tree, alternative NN topologies, and a **1D convolutional neural network**.

> Course project — Deree, The American College of Greece · ITC 4568 Machine Learning · Spring 2026
> Author: Evangelos Passas

## Problem

A vibration sensor is attached to a water pipe with 4 taps. Each reading is labelled by
which tap is open (5 classes):

| Label | State |
|-------|-------|
| 0 | AllTapsClosed |
| 1 | Tap1Open |
| 2 | Tap2Open |
| 3 | Tap3Open |
| 4 | Tap4Open |

The goal is to predict the tap state from the vibration signal.

## Pipeline

The scripts in [`src/`](src/) implement the project in order:

| Step | Script | What it does |
|------|--------|--------------|
| **1A** | `part1a.py` | Reads all per-tap CSVs under a sensor directory → single `ALL_DATA_<X>.CSV` (`ts,value,label`) with a monotonically increasing timeline. |
| **1B** | `part1b.py` | Aggregates readings into **0.1 s windows** (avg/min/max per window, never crossing a label boundary) → `AGGR_DATA_<X>.csv`. |
| **1C** | `part1c.py` | Fuses the two sensors by overlapping time intervals → `AGGR_DATA_ALL.csv` (8 columns), `np.nan` where a sensor has no overlapping reading. |
| **2A** | `part2a.py` | Drops the timestamp, **40-30-30** train/val/test split, trains a shallow MLP (10 hidden, ReLU) and a Decision Tree; reports accuracy & weighted F1. |
| **3A** | `part3a.py` | **Feature engineering** — rolling max-abs / std / mean over the previous 10 s, sensor ranges & differences — then re-runs both models. |
| **3B** | `part3b.py` | Alternative topologies (MLP 64-32 tanh, MLP 128 sigmoid) and a **PyTorch 1D-CNN**. |

## Results (weighted F1 / accuracy on held-out data)

| Model | Features | Test accuracy |
|-------|----------|---------------|
| Shallow MLP (10, ReLU) | raw aggregates | 38.3% |
| Decision Tree | raw aggregates | 33.6% |
| Shallow MLP (10, ReLU) | **engineered** | 72.9% |
| Decision Tree | **engineered** | **96.0%** (eval) |
| MLP (64, 32), tanh | engineered | 89.4% |
| MLP (128), sigmoid | engineered | 58.4% |
| 1D-CNN (PyTorch) | engineered | best of the NN models |

**Takeaway:** raw aggregated statistics are not separable; the temporal **feature
engineering in Part 3A** is what makes the problem learnable, after which a Decision
Tree is remarkably strong and the CNN is the best neural model.

## Setup & running

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate     Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Part 1 (the raw dataset is **not** in this repo — see below):

```bash
python src/part1a.py <Sensor1_directory>      # -> ALL_DATA_1.CSV
python src/part1b.py ALL_DATA_1.CSV           # -> AGGR_DATA_1.csv
python src/part1c.py AGGR_DATA_1.csv AGGR_DATA_2.csv AGGR_DATA_ALL.csv
```

Parts 2 & 3 read `AGGR_DATA_ALL.csv`. A small sample of the aggregated data is included
under [`src/data/`](src/data/) so these scripts can be run without the full dataset —
set the `Path` variable at the top of `part2a.py` / `part3a.py` / `part3b.py` to point
at `src/data/AGGR_DATA_ALL.csv`.

## Data

The raw dataset (~260 MB of per-sensor vibration CSVs, provided via the course) and the
large intermediate `ALL_DATA_*.CSV` files (~208 MB each) are **git-ignored**. Only the
small aggregated CSVs are committed as a runnable sample.

## Notes

- **Part 2A output activation:** the brief asks for sigmoid output nodes; scikit-learn's
  `MLPClassifier` uses softmax for multiclass internally and does not expose the output
  activation, so this is handled by the framework rather than set explicitly.
- The written report (findings, figures, conclusions) is submitted separately on
  Blackboard and is not included here.
