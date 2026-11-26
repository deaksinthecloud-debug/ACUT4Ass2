
## Project Overview
Predicts whether an adult patient will be readmitted to hospital within 30 days,
scored at the point of discharge, so a care-coordination team can prioritise
follow-up calls and critical care intrventions. Binary classification; one prediction per patient discharge.

## Project structure

```
config/config.yaml        All settings: paths, training thresholds, variables and inputs
data/                     Raw dataset and an example file of new patients with my test data sets
src/ingest.py             Load config and data
src/validate.py           Data-quality checks
src/features.py           Engineered features + preprocessing pipeline
src/train.py              dat4a splits, run recording
src/predict.py            prediction for new patients using the approved model
tests/test_pipeline.py    Automated tests (pytest)
jupyter_notebooks/        notebooks for data experimentation
.github/workflows/ci.yml  GitHub Actions: tests, validation and training on every push
artefacts/                validation reports and run records
models/                   Created by the approval gate: the approved model + model card
```

## Set-up

Requires Python 3.13 and Git.

```bash
git clone https://github.com/deaksinthecloud-debug/ACUT4Ass2.git
cd ACUT4Ass2
python -m venv .venv
```

Activate the environment:

- Windows `.venv\Scripts\activate`

Then install the pinned dependencies:

```bash
pip install -r requirements.txt
```

## Data

`data/synthetic_clinical_dataset.csv` – 10,000 synthetic patients, 14 columns.

Dataset version (SHA-256, first 12 characters): `bee99ea6e73b`
Every training run records this hash; if the file changes, the hash changes.

## How to run

Run all commands from the project root, with the environment activated.

| Step | Command | Output |
|---|---|---|
| Validate the data | `python -m src.validate` | `artefacts/validation_report.json` (exit code 1 on failure) |
| Run the tests | `python -m pytest -q` | 12 automated tests |
| Train and evaluate | `python -m src.train` | `artefacts/runs/<run_id>/run.json` and `model.joblib`; `models/model.joblib` if approved |
| Predict new patients | `python -m src.predict --input data/new_patients_example.csv` | `artefacts/predictions.csv` |

Validate a different file: `python -m src.validate --input path/to/file.csv`
(add `--mode serving` for new patients without outcome columns).

Commit before training: each run records the current Git commit, so commit any code or
config changes first and the run record will point to the exact code that produced it.
Note: This caused me some challenges during the early build phase.  An early untracked run selected logistical regression - ROC-AUC 0.630)
and every run since has been random forest at 0.620  this is documented in section 3.4  Remember to put CI/CD processes in place first!

## Results (approved run `20261126T031108Z`, commit `c3ed408`)

| Model | Validation ROC-AUC | Test ROC-AUC | Test recall | Test precision |
|---|---|---|---|---|
| Baseline (dummy) | 0.500 | 0.500 | 0.00 | – |
| Logistic regression | 0.585 | – | – | – |
| Random forest (selected) | 0.590 | 0.620 | 0.60 | 0.21 |

The two candidates performed almost identically on validation; the random forest was
selected on validation ROC-AUC and evaluated once on the held-out test set.

At the 0.5 threshold the model flagged 776 of 1,960 test patients and identified 164 of the
274 readmissions. Random selection of the same number of patients would be expected to reach
about 108, so the model reaches about 51% more at-risk patients for the same workload
(lift 1.51). Full details, including subgroup results, are in the run record.

## Tracing a model back to its evidence

1. Open `models/model_card.json` and note the `run_id`.
2. Open `artefacts/runs/<run_id>/run.json`. It records:
   - code `git_commit` – check out with `git checkout <commit>`
   - data: `dataset_sha256`
   - settings: the full `config`
   - environment Python and library versions
   - results: validation and test metrics, subgroups, approval decision
3. `artefacts/runs_log.jsonl` has one line per run for quick comparison.

## Key design decisions

adults only. 204 patients aged 0–17 are flagged by validation and excluded from training;
out-of-scope or implausible records are flagged for manual review at prediction time.
Leakage controls patient_id`, `mortality` and the target are excluded (reasons in `config.yaml`);
  preprocessing is fitted on the training split only, inside one saved pipeline.
Evaluation: stratified 64/16/20 train/validation/test split; model chosen on validation,
test used once. ROC-AUC for selection; recall and precision at the decision threshold.
Approval gate: a model is promoted to `models/` only if its test ROC-AUC beats the
  baseline by at least 0.05. In real use this would be followed by human clinical sign-off.
Train–serve consistency `src/predict.py` loads the same saved pipeline used in evaluation,
  so feature logic cannot differ between training and prediction (verified by automated tests).

## Limitations

Synthetic data; no admission dates  the definition of `mortality` is undocumented; 
no ethnicity, Aboriginal and Torres Strait Islander status or location data, so fairness across those groups
cannot be assessed. See the technical report.
