import json
from pathlib import Path

import pandas as pd

from src.ingest import load_config, load_raw


def validate(df, cfg):

    errors = []      # problems that must stop the pipeline
    warnings = []    # suspicious things to log, but carry on
    row_bad = pd.Series(False, index=df.index)   # one True/False per row, all False to start
     # ---- Check 1: missing columns (error) ----
    s = cfg["schema"]
    expected = (
        [cfg["data"]["id_column"], cfg["data"]["target"]]
        + list(s["numeric"])
        + s["binary"]
        + list(s["categorical"])
    )

    missing = [c for c in expected if c not in df.columns]

    if missing:
        errors.append(f"missing columns: {missing}")
        return {"passed": False, "n_rows": len(df),
                "errors": errors, "warnings": warnings}

    # ---- Domain check: died AND readmitted (warning) this does make sense as you dont get readmitted when you die ----
    if "mortality" in df.columns:
        both = ((df["mortality"] == 1) & (df[cfg["data"]["target"]] == 1)).sum()
        if both > 0:
            warnings.append(
                f"{both} rows have mortality=1 AND readmission=1 - "
                "label definitions need checking with the data owner medical staff"
            )
   
    return {
        "passed": len(errors) == 0,
        "n_rows": len(df),
        "errors": errors,
        "warnings": warnings,
       }


if __name__ == "__main__":
    cfg = load_config()
    df = load_raw(cfg["data"]["raw_path"])
    report = validate(df, cfg)

    # Keep the file short: save the COUNT of bad rows, not the full list
    to_save = {k: v for k, v in report.items() if k != "bad_rows"}
    to_save["n_bad_rows"] = len(report.get("bad_rows", []))

    Path("artefacts").mkdir(exist_ok=True)              # create folder if missing
    with open("artefacts/validation_report.json", "w") as f:
        json.dump(to_save, f, indent=2)

    print(json.dumps(to_save, indent=2))
    raise SystemExit(0 if report["passed"] else 1)