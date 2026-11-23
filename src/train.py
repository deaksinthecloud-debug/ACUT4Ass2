"""Training: load, validate, prepare, split, train, evaluate and save models."""
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.features import RAW_INPUTS, build_preprocessor
from src.ingest import load_config, load_raw
from src.validate import validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def evaluate(pipe, X, y, threshold=0.5):
    """Score a fitted pipeline on X, y. Returns a dictionary of metrics."""
    risk = pipe.predict_proba(X)[:, 1]          # probability of readmission, per patient
    flagged = (risk >= threshold).astype(int)   # 1 = flag for follow-up

    tn, fp, fn, tp = confusion_matrix(y, flagged, labels=[0, 1]).ravel()

    return {
        "roc_auc": round(roc_auc_score(y, risk), 3),                          # ranking quality
        "recall": round(recall_score(y, flagged, zero_division=0), 3),        # share of readmissions caught
        "precision": round(precision_score(y, flagged, zero_division=0), 3),  # share of flags that were right
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }

def main():
    # 1. Load the settings and the raw data
    cfg = load_config()
    target = cfg["data"]["target"]
    df = load_raw(cfg["data"]["raw_path"])
    print(f"Loaded {len(df):,} rows")

    # 2. Validate - stop if the data is structurally broken
    report = validate(df, cfg, mode="training")
    if not report["passed"]:
        raise SystemExit(f"Validation failed - not training: {report['errors']}")
    for w in report["warnings"]:
        print("WARNING:", w)

    # 3. Remove the rows validation flagged (out of scope, e.g. under-18s)
    df = df.drop(index=report["bad_rows"])
    print(f"Removed {len(report['bad_rows']):,} flagged rows -> {len(df):,} rows remain")

    # 4. Separate inputs (X) from the answer (y)
    X = df[RAW_INPUTS]
    y = df[target]
    print(f"X: {X.shape[0]:,} patients x {X.shape[1]} inputs")
    print(f"y: readmission rate {y.mean():.1%}")

        # 5. Split three ways: train (learn) / validation (choose) / test (final check)
    seed = cfg["random_seed"]

    # First split: hold back the TEST set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=cfg["split"]["test_size"],
        stratify=y,
        random_state=seed,
    )

    # Second split: divide the rest into TRAIN and VALIDATION
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=cfg["split"]["validation_size"],
        stratify=y_temp,
        random_state=seed,
    )

    for name, ys in [("train", y_train), ("validation", y_val), ("test", y_test)]:
        print(f"{name:10s} {len(ys):5,} patients   readmission rate {ys.mean():.1%}")

    # 6. Train the baseline and the candidates, all with the SAME preprocessing
    threshold = cfg["decision_threshold"]
    m = cfg["models"]
    models = {
        "baseline_dummy": DummyClassifier(strategy="prior"),
        "logistic_regression": LogisticRegression(**m["logistic_regression"], random_state=seed),
        "random_forest": RandomForestClassifier(**m["random_forest"], random_state=seed, n_jobs=-1),
    }

    results = {}     # metrics for each model
    fitted = {}      # the trained pipelines, kept for Task 4.7
    print()
    for name, model in models.items():
        pipe = Pipeline([
            ("preprocess", build_preprocessor()),
            ("model", model),
        ])
        pipe.fit(X_train, y_train)                         # learn from TRAIN only
        results[name] = {"validation": evaluate(pipe, X_val, y_val, threshold)}
        fitted[name] = pipe

        v = results[name]["validation"]
        print(f"{name:20s} ROC-AUC {v['roc_auc']:.3f}   recall {v['recall']:.2f}   "
              f"precision {v['precision']:.2f}   flagged {v['tp'] + v['fp']:4d}")

if __name__ == "__main__":
    main()