# 30-day readmission risk – MLOps pipeline

Predicts whether an adult patient will be readmitted to hospital within 30 days,
scored **at the point of discharge**, so a care-coordination team can prioritise
follow-up calls. Binary classification; one prediction per patient discharge.

> The model supports clinical judgement – it does not replace it.

## Project structure

```
config/config.yaml        All settings: paths, schema, split, models, thresholds, approval rule
data/                     Raw dataset (synthetic_clinical_dataset.csv)
src/ingest.py             Load config and data; dataset hash (version)
src/validate.py           Data-quality checks (training and serving modes)
src/features.py           Engineered features + preprocessing pipeline
src/train.py              Split, baseline + candidates, evaluation, run record, approval gate
tests/test_pipeline.py    Automated tests (pytest)
jupyter_notebooks/        Exploration and feature analysis
.github/workflows/ci.yml  GitHub Actions: tests, validation and training on every push
artefacts/                Created by runs: validation report, run records, run log
models/                   Created by the approval gate: the approved model + model card
```

## Set-up

Requires **Python 3.13** and Git.

```bash
git clone
