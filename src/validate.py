import json
from pathlib import Path

import pandas as pd

from src.ingest import load_config, load_raw


def validate(df, cfg):

    errors = []      # problems that must stop the pipeline
    warnings = []    # suspicious things to log, but carry on

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
    print(json.dumps(report, indent=2))   # print the report neatly